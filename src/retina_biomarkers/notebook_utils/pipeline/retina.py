import numpy as np, torch
from PIL import Image

from src.retina_biomarkers.notebook_utils.pipeline.config import PipelineConfig
from src.data.preprocessing import _iso_resize_and_pad
from apps.streamlit.lib.preprocess import preprocess_image_retina_from_pil
from src.retina_biomarkers.notebook_utils.models.mathfi_loader import load_dpcn_from_ckpt, infer_seg_maps

# OD segmentation 
from src.retina_biomarkers.od_seg import (
    load_refuge_segformer, infer_label_map, extract_disc_mask_safe,
    center_and_pd_with_bounds,
)

# Biomarkers
from src.retina_biomarkers.geometry import distance_transform, sample_widths_orthogonal, sample_width_along_skeleton, skeletonize_mask, build_skeleton_graph
from src.retina_biomarkers.metrics_global import (
    area_density, length_density, caliber_stats, tortuosity_stats, fractal_dimension_boxcount
)
from src.retina_biomarkers.regional import metrics_by_rings

def compute_biomarkers_from_mask_array(
    mask_01: np.ndarray,
    *, disc_center=None, PD_px=None,
    max_gap_px: int = 12,
    angle_k_ahead: int = 3,
    ortho_step: float = 0.5,
    ortho_max_radius: float = 20.0,
) -> dict:
    """
    Keeps logic; adds PD-normalized tortuosity (×1e3) to mirror papers.
    Assumes metrics_by_rings() in regional.py accepts edt_interior=... (set default there to be safe).
    """
    mask = (mask_01 > 0).astype(np.uint8)
    H, W = mask.shape

    skel  = skeletonize_mask(mask)
    dist  = distance_transform(mask)
    graph = build_skeleton_graph(skel)

    widths_edt  = sample_width_along_skeleton(dist, graph)
    widths_orth = sample_widths_orthogonal(mask, graph, k_tangent=3, step=ortho_step, max_radius=ortho_max_radius)

    ld_px_inv = float(length_density(graph, mask.shape))
    vc_edt  = caliber_stats(widths_edt)
    vc_orth = caliber_stats(widths_orth)

    g = {
        "area_density":          float(area_density(mask)),
        "length_density_px_inv": ld_px_inv,
        "fractal_dimension":     float(fractal_dimension_boxcount(skel)),
        **tortuosity_stats(graph),
        "vc_edt":  vc_edt,
        "vc_orth": vc_orth,
        "median_width": vc_orth["median_width"],
        "iqr_width":    vc_orth["iqr_width"],
    }
    if PD_px is not None:
        g["median_width_PD"]       = g["median_width"] / PD_px
        g["iqr_width_PD"]          = g["iqr_width"]    / PD_px
        g["length_density_PD_inv"] = ld_px_inv * (PD_px ** 2)
        # tortuosity normalization (px^-2 × PD^2 → dimensionless; present as ×10^-3)
        T_PD2 = float(g["tortuosity_mean"]) * (PD_px ** 2)
        g["tortuosity_mean_PD2"] = T_PD2
        g["tortuosity_mean_PD2_x1e3"] = 1e3 * T_PD2

    # topology 
    from src.retina_biomarkers.metrics_topology import (
        junction_metrics, branching_and_bifurcation_angles, branching_angles_roi, gap_metrics
    )
    topo = {}
    topo.update(junction_metrics(graph, mask.shape))
    topo.update(branching_and_bifurcation_angles(graph, k_ahead=angle_k_ahead))
    topo.update(gap_metrics(mask, graph, max_gap_px=max_gap_px))
    topo["angles_2PD"] = None
    if disc_center is not None and PD_px is not None:
        topo["angles_2PD"] = branching_angles_roi(
            graph, disc_center=disc_center, PD_px=PD_px, max_PD=2.0, k_ahead=angle_k_ahead
        )

    rings = None
    if disc_center is not None and PD_px is not None:
        # IMPORTANT: ensure regional.metrics_by_rings has edt_interior arg or aligns to widths_orth indexing
        rings = metrics_by_rings(
            mask, graph, widths_orth, disc_center=disc_center, PD_px=PD_px,
            use_orth=True
        )

        # add ring tortuosity ×1e3 per PD² if  metrics_by_rings returns px units
        for r in rings.values():
            if "tortuosity_mean" in r and PD_px:
                T_PD2 = float(r["tortuosity_mean"]) * (PD_px ** 2)
                r["tortuosity_mean_PD2_x1e3"] = 1e3 * T_PD2

    return {"image_shape": (H, W), "global": g, "topology": topo, "rings": rings}

def run_pipeline(
    image_path: str,
    ckpt_path: str,
    *,
    fov_path: str | None = None,
    cfg: PipelineConfig = PipelineConfig(),
):
    """End-to-end: image → OD → preprocess (+FOV) → model → masks → biomarkers."""
    # 1) Load fundus + make ISO canvas for OD geometry
    fundus_pil = Image.open(image_path).convert("RGB")
    rgb_iso = _iso_resize_and_pad(np.array(fundus_pil), target=cfg.image_size, pad_value=0).astype(np.uint8)
    H, W = rgb_iso.shape[:2]

    # 2) Optic disc segmentation + PD
    processor, od_model, od_dev = load_refuge_segformer()
    pred_map  = infer_label_map(rgb_iso, processor, od_model, device=od_dev)
    disc_mask = extract_disc_mask_safe(pred_map, od_model.config.id2label, img_shape=rgb_iso.shape,
                                       cup_dilate_frac=cfg.cup_dilate_frac)
    center_yx, PD_raw, PD_px = center_and_pd_with_bounds(
        disc_mask, rgb_iso.shape, r_frac=cfg.r_frac,
        allow_fallback=True, fallback_center_yx=(H/2.0, W/2.0),
        fallback_PD_px=0.20 * min(H, W)
    )

    # 3) Preprocess grayscale for vessel model + FOV
    img_1hw = preprocess_image_retina_from_pil(
        fundus_pil, target_size=cfg.image_size,
        use_gamma=cfg.use_gamma, gamma=cfg.gamma,
        clahe_clip=cfg.clahe_clip, clahe_tiles=cfg.clahe_tiles
    ).astype(np.float32)  # (1,H,W)
    if fov_path is not None:
        from PIL import Image as _PIL
        fov = _PIL.open(fov_path).convert("L").resize((W, H))
        fov_1hw = (np.array(fov) > 0).astype(np.float32)[None]
    else:
        fov_1hw = (img_1hw > 0).astype(np.float32)
    img_fov_1hw = (img_1hw * fov_1hw).astype(np.float32)

    # 4) Model inference
    model, dev = load_dpcn_from_ckpt(ckpt_path)
    x   = torch.from_numpy(img_fov_1hw).unsqueeze(0).to(dev)
    fov = torch.from_numpy(fov_1hw).unsqueeze(0).to(dev)
    mask_u8, probs, logits, edge_probs, skel_probs = infer_seg_maps(
        model, x, fov=fov if cfg.use_fov_in_model else (fov if fov is not None else None),
        use_fov_in_model=cfg.use_fov_in_model, threshold=cfg.threshold
    )

    # 5) Biomarkers
    biom = compute_biomarkers_from_mask_array(
        mask_u8, disc_center=center_yx, PD_px=float(PD_px),
        max_gap_px=cfg.max_gap_px, angle_k_ahead=cfg.angle_k_ahead,
        ortho_step=cfg.ortho_step, ortho_max_radius=cfg.ortho_max_radius
    )

    # Convenience outputs for visuals
    raw_rgb     = np.array(fundus_pil, dtype=np.uint8)
    pre_gray_u8 = (img_1hw[0] * 255.0).astype(np.uint8)
    prob_map    = probs[0, 0].detach().cpu().numpy().astype(np.float32)
    mask_thr05  = (prob_map >= 0.5).astype(np.uint8)

    return {
        "od": {"center_yx": (float(center_yx[0]), float(center_yx[1])),
               "PD_px_raw": float(PD_raw), "PD_px": float(PD_px)},
        "biomarkers": biom,
        "raw_rgb": raw_rgb,
        "rgb_iso": rgb_iso,
        "disc_mask": disc_mask,
        "pre_gray_u8": pre_gray_u8,
        "prob_map": prob_map,
        "pred_mask": mask_u8,
        "pred_mask_thr05": mask_thr05
    }
