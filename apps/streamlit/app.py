# apps/streamlit/app.py
import io, time, contextlib
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# -- project root on sys.path so src.* works
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.training.metrics import (
    dice, iou
)

# --- external deps (optional) ---
try:
    import psutil
except Exception:
    psutil = None
try:
    import torch
except Exception:
    torch = None

# --- app-local helpers (now in lib/) ---
from apps.streamlit.lib.config import (
    DATASET_CHECKPOINTS,
    IMAGE_SIZE_BY_DATASET,
    USE_FOV_IN_MODEL,
)
from apps.streamlit.lib.state import (
    init_state, add_msg, stem_of, pil_from_upload,
    delete_image_by_stem, clear_session_outputs,
)
from apps.streamlit.lib.preprocess import (
    preprocess_image_retina_from_pil,
    preprocess_mask_from_bytes,
    load_fov_1hw_from_bytes,
    fov_bin_from_bytes,
)
from apps.streamlit.lib.model import load_seg_model
from apps.streamlit.lib.ui import (
    try_zoomable, caption_with_size, render_telemetry_sidebar_footer,
    dataset_toggle_row, stage_runner,
)
from apps.streamlit.lib.metrics_ui import (
    compute_metrics_single, render_metric_cards_main, render_metric_cards_others
)
# used only to make overlay base the exact model geometry
from src.data.preprocessing import _iso_resize_and_pad

# --------------------- File Upload Helpers --------------------
# --- Utility: make unique names like "img.png", "img (1).png", "img (2).png" ---
def make_unique_name(name: str, already: set[str]) -> str:
    p = Path(name)
    base, ext = p.stem, p.suffix  # ext includes the dot
    candidate = f"{base}{ext}"
    idx = 1
    while candidate in already:
        candidate = f"{base} ({idx}){ext}"
        idx += 1
    return candidate

# --- Proxy so we can override .name but keep file-like behavior for PIL/Streamlit ---
class UploadedFileProxy:
    def __init__(self, uf, new_name: str):
        self._uf = uf
        self.name = new_name
        self.type = getattr(uf, "type", None)

    # forward common file-like ops
    def read(self, *a, **kw):  return self._uf.read(*a, **kw)
    def seek(self, *a, **kw):  return self._uf.seek(*a, **kw)
    def tell(self, *a, **kw):  return self._uf.tell(*a, **kw)
    def getvalue(self, *a, **kw): return self._uf.getvalue(*a, **kw)

    # forward anything else transparently
    def __getattr__(self, name): 
        return getattr(self._uf, name)


# ---------------------- Page setup ----------------------
st.set_page_config(page_title="Retinal Vessel Segmentation UI", layout="wide")

def load_css(path: str):
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

CSS_PATH = Path(__file__).with_name("styles") / "app_style.css"
if CSS_PATH.exists():
    load_css(str(CSS_PATH))
else:
    st.warning(f"CSS not found at: {CSS_PATH}")

# ---------------------- State ----------------------
init_state()

# ---------------------- Sidebar ----------------------
top = st.sidebar.container()
footer = st.sidebar.container()

with top:
    st.selectbox("Top Mode", ["Single Model (MATFHI)", "Comparison (UNet vs MATFHI)"],
                 index=0, key="mode_top")
    st.radio("Run Mode", ["Predict Only", "With Ground Truth"], key="submode")

with footer:
    st.markdown("#")
    render_telemetry_sidebar_footer(psutil=psutil, torch=torch)  # <- pass deps
    st.markdown("#")
    st.markdown("### How to use")
    st.write("- Upload a **batch of fundus images** below.")
    st.write("- Use **Prev/Next** to browse; upload a **per-image FOV** on the right.")
    st.write("- Run inference; overlay/timing show in the Viewer.")

# ---------------------- Upload ----------------------
st.markdown("####")
st.markdown("# MATHFI: Multi-scale Adaptive Thresholding with Hierarchical Feature Integration")
st.markdown(f"#### Simulation Mode: `{st.session_state.get("submode")}` & `{st.session_state.get("mode_top")}`")
st.divider()

card_upload_sec = st.container(border=True)
with card_upload_sec:
    st.markdown("## Upload Raw Fundus Image(s) here")
    up1 = st.file_uploader(
        "Fundus images (batch upload)",
        type=["png","jpg","jpeg","tif"],
        accept_multiple_files=True,
        key=f"u1_{st.session_state['uploader_nonce']}"
    )


# Sync uploader → session library (respects deletions & dedupes names)
if up1 is not None:
    # current library (before sync)
    old_items = st.session_state.get("files_img", [])
    old_map = {stem_of(f.name): f for f in old_items}

    new_items: list = []
    new_stems: list = []

    skip_stems = st.session_state.get("deleted_stems", set())

    # names that are already in the widget selection as we build it
    # start with *nothing* because we want the widget list to be the source of truth
    assigned_names: set[str] = set()

    for uf in up1:
        # If user deleted a stem from Selection, ignore exact same stem reappearing
        raw_stem = stem_of(uf.name)
        if raw_stem in skip_stems:
            continue

        # Create a unique filename among the *current uploader list* we are building
        unique_name = make_unique_name(uf.name, assigned_names)
        assigned_names.add(unique_name)

        # Keep track of stems for removal detection
        new_stems.append(stem_of(unique_name))

        # Store a proxy with overridden .name so the rest of the app uses the unique name
        new_items.append(UploadedFileProxy(uf, unique_name))

    # Anything that existed but is no longer in the uploader list is truly removed
    removed_stems = set(old_map.keys()) - set(new_stems)
    for s in removed_stems:
        st.session_state["fov_by_stem"].pop(s, None)
        st.session_state["gt_by_stem"].pop(s, None)
        st.session_state["results"].pop(s, None)
        if st.session_state.get("selected_stem") == s:
            st.session_state["selected_stem"] = None

    # Commit the exact list, and keep pagination/index valid
    st.session_state["files_img"] = new_items
    n = len(new_items)
    st.session_state["sel_idx"] = 0 if n == 0 else min(st.session_state.get("sel_idx", 0), n - 1)

img_files = st.session_state["files_img"]

# ---------------------- Selection ----------------------
if img_files:
    stems = [stem_of(f.name) for f in img_files]
    n = len(stems)
    st.session_state["sel_idx"] = max(0, min(st.session_state.get("sel_idx", 0), n - 1))
    idx = st.session_state["sel_idx"]

    st.divider()
    st.markdown("## Preview and Selection")
    c_prev, c_mid, c_next = st.columns([1, 6, 1])
    with c_prev:
        if st.button("◀ Prev", use_container_width=True, disabled=(idx <= 0)):
            st.session_state["sel_idx"] = max(0, idx - 1); st.rerun()
    with c_mid:
        st.write(f"Image {idx+1}/{n}")
    with c_next:
        if st.button("Next ▶", use_container_width=True, disabled=(idx >= n - 1)):
            st.session_state["sel_idx"] = min(n - 1, idx + 1); st.rerun()

    stem = stems[idx]
    file_obj = img_files[idx]
    img = Image.open(file_obj).convert("RGB")
    is_selected = (st.session_state.get("selected_stem") == stem)

    card = st.container(border=True)
    with card:
        top_cols = st.columns([2, 1])
        with top_cols[0]:
            st.markdown(f"**{file_obj.name}**")
            st.image(img, use_container_width=True)

        with top_cols[1]:
            banner_txt = "✅ SELECTED" if is_selected else "NOT SELECTED"
            banner_col = "#0da2a2" if is_selected else "#c45959"
            st.markdown(
                f"<div style='font-size:1.35rem;font-weight:500;color:{banner_col};"
                f"margin:.25rem 0 .5rem 0'>{banner_txt}</div>",
                unsafe_allow_html=True
            )

            nonce = st.session_state["fov_uploader_nonce"].get(stem, 0)
            st.markdown("##### Upload FOV (Field of View)")
            fov_up = st.file_uploader(
                f"FOV for {stem}",
                type=["png","jpg","jpeg","tif"],
                key=f"fov_sel_{stem}_{nonce}"
            )
            if fov_up is not None:
                st.session_state["fov_by_stem"][stem] = {
                    "name": fov_up.name,
                    "mime": fov_up.type or "image/png",
                    "bytes": fov_up.getvalue(),
                }
                st.session_state["fov_uploader_nonce"][stem] = nonce + 1
                st.rerun()

            fov_entry = st.session_state["fov_by_stem"].get(stem)
            if fov_entry:
                st.image(Image.open(io.BytesIO(fov_entry["bytes"])), use_container_width=True)
                if st.button("Remove FOV", key=f"rm_fov_sel_{stem}", use_container_width=True):
                    st.session_state["fov_by_stem"].pop(stem, None)
                    st.session_state["fov_uploader_nonce"][stem] = st.session_state["fov_uploader_nonce"].get(stem, 0) + 1
                    st.rerun()
            else:
                st.caption("No FOV paired yet.")

            st.divider()
            # --- Ground Truth uploader (only show in "With Ground Truth" mode) ---
            if st.session_state.get("submode") == "With Ground Truth":
                st.markdown("##### Upload Ground Truth Image")

                gt_nonce = st.session_state["gt_uploader_nonce"].get(stem, 0)
                gt_up = st.file_uploader(
                    f"GT for {stem}",
                    type=["png", "jpg", "jpeg", "tif"],
                    key=f"gt_sel_{stem}_{gt_nonce}"
                )
                if gt_up is not None:
                    st.session_state["gt_by_stem"][stem] = {
                        "name": gt_up.name,
                        "mime": gt_up.type or "image/png",
                        "bytes": gt_up.getvalue(),
                    }
                    st.session_state["gt_uploader_nonce"][stem] = gt_nonce + 1
                    st.rerun()

                gt_entry = st.session_state["gt_by_stem"].get(stem)
                if gt_entry:
                    st.image(Image.open(io.BytesIO(gt_entry["bytes"])), use_container_width=True)
                    if st.button("Remove GT", key=f"rm_gt_sel_{stem}", use_container_width=True):
                        st.session_state["gt_by_stem"].pop(stem, None)
                        st.session_state["gt_uploader_nonce"][stem] = st.session_state["gt_uploader_nonce"].get(stem, 0) + 1
                        st.rerun()
                else:
                    st.caption("No Ground Truth paired yet.")


        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            if st.button("Select this image", key=f"select_{stem}", use_container_width=True, disabled=is_selected):
                st.session_state["selected_stem"] = stem; st.rerun()
        with c_sel2:
            if st.button("Unselect", key=f"unselect_{stem}", use_container_width=True, disabled=not is_selected):
                st.session_state["selected_stem"] = None; st.rerun()

        if st.button("Delete Image / Image Pair", key=f"del_img_{stem}", use_container_width=True):
            delete_image_by_stem(stem)

# ---------------------- Viewer & status ----------------------
st.divider()
viewer = st.container()
with viewer:
    sel_stem = st.session_state.get("selected_stem")
    has_result = bool(sel_stem and st.session_state.get("results", {}).get(sel_stem))
    has_selected_file = bool(sel_stem and any(stem_of(f.name) == sel_stem for f in img_files))

    st.markdown("## Inference")
    fov_entry_hdr = st.session_state.get("fov_by_stem", {}).get(sel_stem)
    has_fov_for_sel = bool(fov_entry_hdr and fov_entry_hdr.get("bytes"))

    header_cols = st.columns([1, 1, 1])
    with header_cols[0]:
        raw_header_cols = st.columns([3, 1])
        with raw_header_cols[0]:
            st.markdown("#### Selected Raw Image")
        with raw_header_cols[1]:
            if has_fov_for_sel:
                st.toggle("FOV", key="fov_tog")
            else:
                st.session_state.pop("fov_tog", None)
    with header_cols[1]:
        st.markdown("#### Preprocessed Image")
    with header_cols[2]:
        predicted_header_vessel_map_cols = st.columns([2, 1])
        with predicted_header_vessel_map_cols[0]:
            st.markdown("#### Predicted Vessel Map")
        with predicted_header_vessel_map_cols[1]:
            if st.session_state.pop("overlay_reset", False):
                st.session_state.pop("overlay_tog", None)
                st.session_state.pop("alpha", None)
            st.toggle("Overlay", key="overlay_tog", disabled=not has_result)

    stage = st.empty()
    img_col, prob_col, out_col = st.columns([1, 1, 1])

    if sel_stem:
        img_file = next((f for f in img_files if stem_of(f.name) == sel_stem), None)
        if img_file is None:
            st.warning("Selected image not found.")
        else:
            img = pil_from_upload(img_file)

            # Raw + FOV overlay
            with img_col:
                base_pil = img
                W0, H0 = base_pil.size
                fov_entry = st.session_state.get("fov_by_stem", {}).get(sel_stem)
                if fov_entry and st.session_state.get("fov_tog", False):
                    fov01 = fov_bin_from_bytes(fov_entry["bytes"], (W0, H0))
                    base_rgb = np.array(base_pil, dtype=np.uint8)
                    base_rgb[fov01 == 0] = 0
                    show_pil = Image.fromarray(base_rgb)
                    try_zoomable(caption_with_size("Original (FOV applied)", show_pil), show_pil)
                else:
                    try_zoomable(caption_with_size("Original (Raw image input)", base_pil), base_pil)

            # Results panes
            res = st.session_state["results"].get(sel_stem)
            if res:
                with prob_col:
                    pre_img = res.get("pre")
                    if pre_img is not None:
                        st.image(pre_img, caption="Preprocessed Image | 512 × 512px |", use_container_width=True, clamp=True)
                    else:
                        st.warning("No preprocessed image saved.")

                with out_col:
                    overlay_on = st.session_state.get("overlay_tog", False)
                    if overlay_on:
                        thr = st.session_state.get("threshold", 0.5)
                        prob_np = res["probs"]
                        mask_bin = (prob_np >= thr).astype(np.uint8) * 255

                        alpha_pct = st.session_state.get("alpha", 50)
                        alpha = alpha_pct / 100.0

                        base_rgb = res.get("overlay_base_rgb")
                        if base_rgb is None:
                            base_rgb = _iso_resize_and_pad(
                                np.array(img.convert("RGB")), target=IMAGE_SIZE_BY_DATASET.get(st.session_state["dataset_choice"], 512), pad_value=0
                            ).astype(np.uint8)

                        base = base_rgb.astype(np.float32)
                        mask01 = (mask_bin.astype(np.float32) / 255.0)[..., None]
                        alpha_map = alpha * mask01
                        blended = np.clip(base * (1.0 - alpha_map) + 255.0 * alpha_map, 0, 255).astype(np.uint8)
                        try_zoomable("Overlay", Image.fromarray(blended))
                        st.slider("Opacity", 0, 100, alpha_pct, key="alpha")
                    else:
                        thr = st.session_state.get("threshold", 0.5)
                        mask_bin = (res["probs"] >= thr).astype(np.uint8) * 255
                        st.image(mask_bin, caption="Predicted Vessel Map (MATHFI)", use_container_width=True)

                st.caption(f"Time: {res['timings']['total_ms']:.1f} ms • Device: {res['device']}")

            else:
                if st.session_state.get("running"):
                    with prob_col:
                        st.info("Preprocessing…")
                    with out_col:
                        st.info("Predicting…")
                else:
                    with prob_col:
                        st.warning("⚠️ Run inference to view preprocessed image.")
                    with out_col:
                        st.warning("⚠️ Run inference to view probability map.")
    else:
        st.info("No image selected. Choose one in the Selection gallery above.")

    # controls
    run_cols = st.columns([1, 1, 1])
    dataset_toggle_row(disabled=st.session_state.get("running", False))
    with run_cols[0]:
        btn_run_viewer = st.button(
            "Run Inference",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("running", False) or not has_selected_file
        )
    with run_cols[1]:
        btn_stop_viewer = st.button(
            "Stop",
            use_container_width=True,
            disabled=(not st.session_state.get("running", False))
        )
    with run_cols[2]:
        btn_clear_viewer = st.button(
            "Clear",
            use_container_width=True,
            disabled=not has_result
        )

    if btn_clear_viewer and sel_stem:
        st.session_state["results"].pop(sel_stem, None)
        st.session_state["overlay_reset"] = True
        st.rerun()

    if btn_stop_viewer:
        st.session_state["stop_flag"] = True
        add_msg("info", "Stop requested; finishing current step…")

# ---------------------- Inference trigger ----------------------
if 'btn_run_viewer' in locals() and btn_run_viewer and has_selected_file and (not st.session_state.get("running", False)):
    sel_stem = st.session_state["selected_stem"]
    img_file = next((f for f in st.session_state["files_img"] if stem_of(f.name) == sel_stem), None)

    if img_file is None:
        add_msg("error", "Selected image not found.")
    else:
        st.session_state["running"] = True
        st.session_state["stop_flag"] = False

        ds = st.session_state.get("dataset_choice", "DRIVE")
        IMAGE_SIZE = IMAGE_SIZE_BY_DATASET.get(ds, 512)

        model, dev, _ = load_seg_model(dataset=ds)

        # 1) Preprocess
        stage_runner(stage, "Preprocessing…"); time.sleep(0.05)
        fundus_pil = pil_from_upload(img_file)
        img_1hw = preprocess_image_retina_from_pil(
            fundus_pil,
            target_size=IMAGE_SIZE,
            use_gamma=True, gamma=0.9,
            clahe_clip=2.0, clahe_tiles=8,
        ).astype(np.float32)

        # 2) FOV
        fov_entry = st.session_state["fov_by_stem"].get(sel_stem)
        if fov_entry and fov_entry.get("bytes"):
            fov_1hw = load_fov_1hw_from_bytes(fov_entry["bytes"], target_size=IMAGE_SIZE)
        else:
            fov_1hw = (img_1hw > 0).astype(np.float32)

        # 3) Apply FOV before model
        img_fov_1hw = (img_1hw * fov_1hw).astype(np.float32)
        pre_img_vis = (img_fov_1hw[0] * 255.0).astype(np.uint8)

        # 4) Inference
        x   = torch.from_numpy(img_fov_1hw).unsqueeze(0).to(dev)
        fov = torch.from_numpy(fov_1hw).unsqueeze(0).to(dev)

        stage_runner(stage, "Predicting…"); time.sleep(0.05)
        t0 = time.time()
        with torch.no_grad():
            use_amp = (dev == "cuda")
            amp_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(x, fov=fov) if USE_FOV_IN_MODEL else model(x)
        probs = torch.sigmoid(logits)
        total_ms = (time.time() - t0) * 1000.0

        # 5) Post
        probs = probs * (fov > 0.5).float()
        thr = st.session_state.get("threshold", 0.5)
        pred01 = (probs >= thr).float()

        mask_bin_u8 = (pred01[0,0].cpu().numpy() * 255).astype(np.uint8)
        prob_np     =  probs[0,0].cpu().numpy().astype(np.float32)

        # 6) Optional metrics
        metrics = None
        if st.session_state.get("submode") == "With Ground Truth":
            gt_entry = st.session_state.get("gt_by_stem", {}).get(sel_stem)
            if gt_entry and gt_entry.get("bytes"):
                gt_1hw = preprocess_mask_from_bytes(gt_entry["bytes"], target_size=IMAGE_SIZE).astype(np.float32)
                y = torch.from_numpy(gt_1hw).unsqueeze(0).to(dev)
                m = (fov > 0.5).float()
                tp = (pred01*m*y).sum().item()
                fn = ((1-pred01)*m*y).sum().item()
                fp = (pred01*m*(1-y)).sum().item()
                dice = (2*tp) / max(1.0, 2*tp + fp + fn)
                iou  = tp / max(1.0, tp + fp + fn)
                metrics = {"dice": float(dice), "iou": float(iou)}

        overlay_base_rgb = _iso_resize_and_pad(
            np.array(fundus_pil.convert("RGB")), target=IMAGE_SIZE, pad_value=0
        ).astype(np.uint8)

        st.session_state["results"][sel_stem] = {
            "probs": prob_np,
            "mask":  mask_bin_u8,
            "pre":   pre_img_vis,
            "overlay_base_rgb": overlay_base_rgb,
            "timings": {"total_ms": total_ms},
            "device": dev,
            "metrics": metrics
        }
        stage_runner(stage, "Done.")
        st.session_state["running"] = False
        st.session_state["done_once"] = True
        st.rerun()

# --- Ground Truth vs Prediction and metrics section -----
if (
    st.session_state.get("submode") == "With Ground Truth"
    and sel_stem
    and has_result
):
    gt_entry = st.session_state.get("gt_by_stem", {}).get(sel_stem)
    if gt_entry and gt_entry.get("bytes"):
        st.divider()
        st.markdown("### Ground Truth vs Prediction")

        col_raw, col_gt, col_pred, col_cmp = st.columns([1, 1, 1, 1])

        # Ensure geometry matches model output
        ds = st.session_state.get("dataset_choice", "DRIVE")
        IMAGE_SIZE = IMAGE_SIZE_BY_DATASET.get(ds, 512)

        # --- 1) GT as (1,H,W) -> vis u8 ---
        gt_1hw = preprocess_mask_from_bytes(gt_entry["bytes"], target_size=IMAGE_SIZE)  # (1,H,W) {0,1}
        gt = (gt_1hw[0] > 0.5).astype(np.uint8)                                         # (H,W) {0,1}
        gt_vis = (gt * 255).astype(np.uint8)

        # --- 2) Prediction from stored probs + current threshold ---
        thr = st.session_state.get("threshold", 0.5)
        prob_np = st.session_state["results"][sel_stem]["probs"]                        # (H,W) float32
        pred = (prob_np >= thr).astype(np.uint8)                                        # (H,W) {0,1}
        pred_vis = (pred * 255).astype(np.uint8)

        with col_raw:
            st.image(pre_img, caption="Preprocessed Image", use_container_width=True, clamp=True)
        with col_gt:
            st.image(gt_vis, caption="Ground Truth", use_container_width=True)

        with col_pred:
            st.image(pred_vis, caption="Predicted Vessel Map (MATHFI)", use_container_width=True)

        with col_cmp:
            # Mutually exclusive difference map
            tp = (pred == 1) & (gt == 1)   # correct vessel
            fn = (pred == 0) & (gt == 1)   # missed vessel (GT only)
            fp = (pred == 1) & (gt == 0)   # over-segmentated (pred only)

            diff_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
            diff_rgb[tp] = [255, 255, 255]          # white (correct vessels)
            diff_rgb[fn] = [255,   0,   0]          # (red missed GT)
            diff_rgb[fp] = [204, 255, 0]          # yellow (over-segmented)

            st.image(
                diff_rgb,
                caption="Comparison Result",
                use_container_width=True
            )


        # --- 3) Metrics (GLOBAL, not FOV-masked) ---
        metrics_all = compute_metrics_single(
            pred_probs=prob_np,      # (H,W) float
            gt_1hw=gt_1hw,           # (1,H,W) {0,1}
            fov_1hw=None,            # no FOV masking
            threshold=thr,
            compute_auc=True,
        )

        prediction_metrics_col = st.columns([1.5,0.2,1.5,0.75,1.25])
        with prediction_metrics_col[0]:
            render_metric_cards_main(metrics_all)
        
        with prediction_metrics_col[2]:
            render_metric_cards_others(metrics_all)

        with prediction_metrics_col[4]:# Tiny legend row
            st.markdown(
                """
                <div style="display:flex; gap:0.5rem; align-items:center; font-size:0.9rem;">
                  <span style="display:inline-block;width:1.5rem;height:0.85rem;background:#ffffff;border:1px solid #888;"></span> Correct (True Postive)
                  <span style="display:inline-block;width:1.5rem;height:0.85;background:#ff0000;border:1px solid #888;"></span> Missed (False Negative)
                  <span style="display:inline-block;width:1.5rem;height:0.85;background:#ccff00;border:1px solid #888;"></span> Over-segmented (False Positive)
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("Upload a Ground Truth mask in the Selection panel to see the comparison and metrics.")


# ---------------------- Comparison scaffold ----------------------
if st.session_state["mode_top"] == "Comparison (UNet vs MATFHI)":
    st.warning("Comparison mode scaffolded. Later: load UNet + MATFHI and render side-by-side with the same inputs/threshold.")
