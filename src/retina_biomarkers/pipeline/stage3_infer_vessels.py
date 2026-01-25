# Stage 3 reads the Stage 1 preprocessed vessel input (img_fov_1hw.npy, fov_1hw.npy) and runs:

# load_dpcn_from_ckpt(ckpt_path) once

# infer_seg_maps(...) per image

# Then it caches probability maps (and optionally logits/edges) so that Stage 4 (biomarkers) becomes pure CPU math on cached arrays.

# src/retina_biomarkers/pipeline/stage3_infer_vessels.py
from __future__ import annotations

import json, hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.retina_biomarkers.notebook_utils.pipeline.config import PipelineConfig
from src.retina_biomarkers.notebook_utils.models.mathfi_loader import load_dpcn_from_ckpt, infer_seg_maps


# ----------------------------
# helpers
# ----------------------------
def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if is_dataclass(cfg):
        return asdict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)}

def safe_image_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")

def stage1_dir(cache_root: str | Path, run_id_stage1: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage1 / "stage1" / safe_image_id(image_id)

def stage3_dir(cache_root: str | Path, run_id_stage3: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage3 / "stage3" / safe_image_id(image_id)

def ckpt_fingerprint(ckpt_path: str | Path) -> str:
    # stable small ID for checkpoint file name + size (fast, no full hash read)
    p = Path(ckpt_path)
    s = f"{p.name}|{p.stat().st_size}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def make_run_id_stage3(cfg: PipelineConfig, ckpt_path: str | Path, prefix: str = "aptos") -> str:
    d = _cfg_to_dict(cfg)
    payload = json.dumps(d, sort_keys=True, default=str) + "|" + str(Path(ckpt_path).name)
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_stage3_{h}"


# ----------------------------
# model context (load once)
# ----------------------------
class VesselContext:
    def __init__(self, ckpt_path: str | Path):
        self.ckpt_path = str(ckpt_path)
        self.model, self.device = load_dpcn_from_ckpt(self.ckpt_path)
        self.model.eval()

    @torch.no_grad()
    def infer_prob_map(
        self,
        img_fov_1hw: np.ndarray,
        fov_1hw: Optional[np.ndarray],
        cfg: PipelineConfig,
    ) -> Dict[str, Any]:
        """
        img_fov_1hw: (1,H,W) float32
        fov_1hw: (1,H,W) float32 (optional)
        returns: dict with prob_map (H,W) float32 + optional extras
        """
        x = torch.from_numpy(img_fov_1hw).unsqueeze(0).to(self.device)  # (1,1,H,W)
        fov = None
        if fov_1hw is not None:
            fov = torch.from_numpy(fov_1hw).unsqueeze(0).to(self.device)  # (1,1,H,W)

        mask_u8, probs, logits, edge_probs, skel_probs = infer_seg_maps(
            self.model,
            x,
            fov=fov if cfg.use_fov_in_model else (fov if fov is not None else None),
            use_fov_in_model=cfg.use_fov_in_model,
            threshold=cfg.threshold,
        )

        prob_map = probs[0, 0].detach().cpu().numpy().astype(np.float32)  # (H,W)
        out = {"prob_map": prob_map}

        # optional: keep outputs for later analysis
        if edge_probs is not None:
            out["edge_prob"] = edge_probs[0, 0].detach().cpu().numpy().astype(np.float32)
        if skel_probs is not None:
            out["skel_prob"] = skel_probs[0, 0].detach().cpu().numpy().astype(np.float32)

        # derived mask (optional convenience)
        out["pred_mask_u8"] = (prob_map >= float(cfg.threshold)).astype(np.uint8)

        return out


# ----------------------------
# stage3 per-image
# ----------------------------
def stage3_infer_one(
    *,
    image_id: str,
    cfg: PipelineConfig,
    ckpt_path: str | Path,
    cache_root: str | Path,
    run_id_stage1: str,
    run_id_stage3: str,
    vessel_ctx: VesselContext,
    save_pred_mask: bool = True,
    prob_dtype: str = "float16",   # "float16" or "float32"
    save_preview_png: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Reads Stage1 artifacts (img_fov_1hw.npy, fov_1hw.npy),
    runs vessel model, and caches prob_map (and optional mask).
    """
    s1 = stage1_dir(cache_root, run_id_stage1, image_id)
    if not s1.exists():
        raise FileNotFoundError(f"Stage1 folder missing for {image_id}: {s1}")

    out_dir = stage3_dir(cache_root, run_id_stage3, image_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = out_dir / "meta.json"
    if done.exists() and not overwrite:
        return {"image_id": image_id, "status": "skipped", "out_dir": str(out_dir)}

    img_fov_1hw = np.load(s1 / "img_fov_1hw.npy").astype(np.float32)   # (1,H,W)
    fov_1hw = np.load(s1 / "fov_1hw.npy").astype(np.float32) if (s1 / "fov_1hw.npy").exists() else None

    pred = vessel_ctx.infer_prob_map(img_fov_1hw, fov_1hw, cfg)

    prob_map = pred["prob_map"]
    if prob_dtype == "float16":
        prob_save = prob_map.astype(np.float16)
    else:
        prob_save = prob_map.astype(np.float32)

    np.save(out_dir / "prob_map.npy", prob_save)

    if save_pred_mask:
        np.save(out_dir / "pred_mask_u8.npy", pred["pred_mask_u8"].astype(np.uint8))

    # optional extras
    if "edge_prob" in pred:
        np.save(out_dir / "edge_prob.npy", pred["edge_prob"].astype(np.float16 if prob_dtype == "float16" else np.float32))
    if "skel_prob" in pred:
        np.save(out_dir / "skel_prob.npy", pred["skel_prob"].astype(np.float16 if prob_dtype == "float16" else np.float32))

    meta = {
        "image_id": image_id,
        "run_id_stage1": run_id_stage1,
        "ckpt_path": str(ckpt_path),
        "ckpt_fingerprint": ckpt_fingerprint(ckpt_path),
        "cfg": _cfg_to_dict(cfg),
        "saved": {
            "prob_map_dtype": prob_dtype,
            "pred_mask_u8": bool(save_pred_mask),
            "edge_prob": bool((out_dir / "edge_prob.npy").exists()),
            "skel_prob": bool((out_dir / "skel_prob.npy").exists()),
        }
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if save_preview_png:
        # quick preview: show prob heat + mask contour on grayscale
        pre_gray = np.load(s1 / "img_1hw.npy")[0]
        mask = pred["pred_mask_u8"].astype(bool)

        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(pre_gray, cmap="gray")
        ax1.set_title("pre_gray")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(prob_map, cmap="gray")
        ax2.contour(mask.astype(float), levels=[0.5], linewidths=0.7)
        ax2.set_title(f"prob + thr@{cfg.threshold}")
        ax2.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / "preview_pred.png", dpi=140)
        plt.close(fig)

    return {"image_id": image_id, "status": "ok", "out_dir": str(out_dir), "prob_min": float(prob_map.min()), "prob_max": float(prob_map.max())}


def stage3_infer_batch(
    items,
    *,
    cfg: PipelineConfig,
    ckpt_path: str | Path,
    cache_root: str | Path,
    run_id_stage1: str,
    run_id_stage3: str,
    overwrite: bool = False,
    save_pred_mask: bool = True,
    prob_dtype: str = "float16",
    save_preview_png: bool = False,
):
    vessel_ctx = VesselContext(ckpt_path)  # load ONCE
    results = []
    for it in items:
        r = stage3_infer_one(
            image_id=it.image_id,
            cfg=cfg,
            ckpt_path=ckpt_path,
            cache_root=cache_root,
            run_id_stage1=run_id_stage1,
            run_id_stage3=run_id_stage3,
            vessel_ctx=vessel_ctx,
            overwrite=overwrite,
            save_pred_mask=save_pred_mask,
            prob_dtype=prob_dtype,
            save_preview_png=save_preview_png,
        )
        results.append(r)
    return results
