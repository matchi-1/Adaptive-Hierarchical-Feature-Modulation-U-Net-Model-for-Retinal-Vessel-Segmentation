# apps/streamlit/app.py
import io, time, pathlib, base64, contextlib
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

# ---------------------- Page setup ----------------------
st.set_page_config(page_title="Retinal Vessel Segmentation UI", layout="wide")

# ---------------------- Load external CSS ----------------------
def load_css(path: str):
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load CSS file at apps/streamlit/styles/app_style.css
CSS_PATH = Path(__file__).with_name("styles") / "app_style.css"
if CSS_PATH.exists():
    load_css(str(CSS_PATH))
else:
    st.warning(f"CSS not found at: {CSS_PATH}")

# ---------------------- Optional deps (safe fallbacks) ----------------------
try:
    import psutil
except Exception:
    psutil = None
try:
    import torch
except Exception:
    torch = None
try:
    # Optional zoom component: pip install streamlit-zoomable-image
    from streamlit_zoomable_image import zoomable_image
except Exception:
    zoomable_image = None

# ---------------------- State ----------------------
def init_state():
    ss = st.session_state
    ss.setdefault("mode_top", "Single Model (MATFHI)")
    ss.setdefault("submode", "Predict Only")  # or "With Ground Truth"
    ss.setdefault("selected_stem", None)
    ss.setdefault("sel_idx", 0)                         # pagination index
    ss.setdefault("files_img", [])                      # uploaded fundus files
    ss.setdefault("fov_by_stem", {})                   # {stem: {"name","mime","bytes"}}
    ss.setdefault("files_gt", [])                      # kept for compatibility (unused here)
    ss.setdefault("files_fov", [])                     # legacy; not used anymore
    ss.setdefault("results", {})                       # stem -> dict(prob, mask, timings, metrics)
    ss.setdefault("messages", [])                      # bottom console messages
    ss.setdefault("running", False)
    ss.setdefault("stop_flag", False)
    ss.setdefault("done_once", False)
init_state()

# ---------------------- Helpers ----------------------
def add_msg(kind: str, text: str):
    st.session_state["messages"].append({"kind": kind, "text": text})

def pil_from_upload(f) -> Image.Image:
    return Image.open(f).convert("RGB")

def to_gray(im: Image.Image) -> Image.Image:
    return im.convert("L")

def stem_of(name: str) -> str:
    return pathlib.Path(name).stem

def colorize_mask(mask_255: np.ndarray) -> np.ndarray:
    """Return an RGB color mask (red) from a single-channel 0..255 mask."""
    h, w = mask_255.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = mask_255  # red channel
    return rgb

def blend(img_rgb_u8: np.ndarray, overlay_rgb_f32: np.ndarray, alpha: float) -> np.ndarray:
    base = img_rgb_u8.astype(np.float32)
    out = (base * (1.0 - alpha) + overlay_rgb_f32 * alpha).clip(0, 255).astype(np.uint8)
    return out

def compute_basic_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, fov_bin: Optional[np.ndarray]):
    m = fov_bin if fov_bin is not None else np.ones_like(gt_bin, dtype=np.uint8)
    P = pred_bin[m == 1]; G = gt_bin[m == 1]
    tp = int(((P == 1) & (G == 1)).sum()); tn = int(((P == 0) & (G == 0)).sum())
    fp = int(((P == 1) & (G == 0)).sum()); fn = int(((P == 0) & (G == 1)).sum())
    total = max(1, tp + tn + fp + fn)
    acc  = (tp + tn) / total
    sen  = tp / max(1, tp + fn)
    spe  = tn / max(1, tn + fp)
    dice = (2 * tp) / max(1, 2 * tp + fp + fn)
    iou  = tp / max(1, tp + fp + fn)
    return dict(acc=acc, sen=sen, spe=spe, dice=dice, f1=dice, iou=iou)

def device_label() -> str:
    if torch is None:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model(model_name: str, device: str = "auto"):
    # TODO: replace with real MATFHI / UNet model loading
    class Dummy:
        name = model_name
        def infer(self, img: Image.Image) -> np.ndarray:
            # Fake prob map [H,W] in [0,1] just to prove UI
            w, h = img.size
            yy, xx = np.mgrid[0:h, 0:w]
            prob = (np.sin(xx / 20.0) * np.cos(yy / 25.0) + 1.0) * 0.5
            cx, cy = w // 2, h // 2
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            prob += np.exp(-(r / (min(w, h) / 3.0)) ** 2) * 0.25
            return np.clip(prob, 0, 1).astype(np.float32)
    return Dummy()

def render_telemetry_sidebar_footer():
    """Device/System details + Notes at the bottom of the sidebar (sticky via CSS)."""
    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    st.caption("Device / System")
    if psutil:
        st.write(f"CPU: {psutil.cpu_percent(interval=None)}%")
    else:
        st.write("CPU: psutil not installed")
    if torch and torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"VRAM: {used:.2f} / {total:.2f} GB")
    else:
        st.write(f"GPU: none (using {device_label()})")

    st.markdown("### How to use")
    st.write("- Upload a **batch of fundus images** below.")
    st.write("- Use **Prev/Next** to browse; upload a **per-image FOV** on the right.")
    st.write("- FOV is automatically **paired** with its image.")
    st.write("- Run inference from the sidebar; overlay/timing show in the Viewer.")
    st.markdown('</div>', unsafe_allow_html=True)

def try_zoomable(label: str, img: Image.Image):
    if zoomable_image:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        zoomable_image(f"data:image/png;base64,{b64}", label=label, height=520)
    else:
        st.image(img, caption=label, use_container_width=True)

def stage_runner(stage_placeholder, text):
    stage_placeholder.markdown(
        f"<div class='status-ribbon'><b>Status:</b> {text}</div>",
        unsafe_allow_html=True
    )

# --- image delete in selection stem ---
def delete_image_by_stem(stem: str):
    """Remove image (and paired FOV/results) by stem, then rerun."""
    st.session_state["files_img"] = [
        f for f in st.session_state.get("files_img", []) if stem_of(f.name) != stem
    ]
    # paired FOV
    st.session_state["fov_by_stem"].pop(stem, None)
    # any results
    st.session_state["results"].pop(stem, None)
    # selection / pagination fixup
    if st.session_state.get("selected_stem") == stem:
        st.session_state["selected_stem"] = None
    n = len(st.session_state.get("files_img", []))
    if n == 0:
        st.session_state["sel_idx"] = 0
    else:
        st.session_state["sel_idx"] = min(st.session_state["sel_idx"], n - 1)
    st.rerun()

def clear_session_outputs():
    st.session_state["results"] = {}
    st.session_state["messages"] = []
    st.session_state["selected_stem"] = None
    st.session_state["running"] = False
    st.session_state["stop_flag"] = False
    st.session_state["done_once"] = False
    st.session_state["sel_idx"] = 0
    st.session_state["fov_by_stem"] = {}

# ---------------------- Sidebar (top controls + sticky footer) ----------------------
top = st.sidebar.container()
footer = st.sidebar.container()  # styled to bottom via CSS

with top:
    model_mode = st.selectbox("Top Mode", ["Single Model (MATFHI)", "Comparison (UNet vs MATFHI)"],
                              index=0, key="mode_top")
    submode = st.radio("Run Mode", ["Predict Only", "With Ground Truth"], key="submode")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
    mask_outside_fov = st.checkbox("Mask outside FOV (metrics)", value=True)
    st.divider()
    btn_run = st.button("Run Inference", type="primary", use_container_width=True)
    btn_stop = st.button("Stop", use_container_width=True, disabled=st.session_state["running"] is False)
    if st.session_state["done_once"]:
        btn_reset = st.button("Reset", use_container_width=True)
    else:
        btn_reset = False

with footer:
    render_telemetry_sidebar_footer()

if btn_reset:
    clear_session_outputs()
    st.rerun()

if btn_stop:
    st.session_state["stop_flag"] = True
    add_msg("info", "Stop requested; finishing current step…")

# ---------------------- Main layout ----------------------
st.markdown("#### Inputs")
up1 = st.file_uploader(
    "Fundus images (batch upload)",
    type=["png","jpg","jpeg","tif"],
    accept_multiple_files=True,
    key="u1"
)

# Keep uploaded files in session (so Reset can clear them)
if up1 is not None:
    st.session_state["files_img"] = up1

img_files = st.session_state["files_img"]

# ---------------------- Selection (one-by-one with pagination + per-image FOV) ----------------------
if img_files:
    stems = [stem_of(f.name) for f in img_files]
    n = len(stems)

    # Bound the current index
    st.session_state["sel_idx"] = max(0, min(st.session_state.get("sel_idx", 0), n - 1))
    idx = st.session_state["sel_idx"]

    st.markdown("#### Selection")

    # Pagination controls
    c_prev, c_mid, c_next = st.columns([1, 6, 1])
    with c_prev:
        if st.button("◀ Prev", use_container_width=True, disabled=(idx <= 0)):
            st.session_state["sel_idx"] = max(0, idx - 1)
            st.rerun()
    with c_mid:
        st.write(f"Image {idx+1}/{n}")
        new_idx = st.slider("Go to", 1, n, idx+1, key="sel_slider", label_visibility="collapsed")
        if new_idx - 1 != idx:
            st.session_state["sel_idx"] = new_idx - 1
            st.rerun()
    with c_next:
        if st.button("Next ▶", use_container_width=True, disabled=(idx >= n - 1)):
            st.session_state["sel_idx"] = min(n - 1, idx + 1)
            st.rerun()

    # Current item card
    stem = stems[idx]
    file_obj = img_files[idx]
    img = Image.open(file_obj).convert("RGB")

    st.divider()
    card = st.container(border=True)
    with card:
        top_cols = st.columns([2, 1])
        with top_cols[0]:
            st.markdown(f"**{file_obj.name}**")
            st.image(img, use_container_width=True)
        with top_cols[1]:
            # Per-image FOV upload
            fov_up = st.file_uploader(f"FOV for {stem}", type=["png","jpg","jpeg","tif"], key=f"fov_{stem}")
            if fov_up is not None:
                st.session_state["fov_by_stem"][stem] = {
                    "name": fov_up.name,
                    "mime": fov_up.type or "image/png",
                    "bytes": fov_up.getvalue(),
                }

            # Show paired FOV if present
            fov_entry = st.session_state["fov_by_stem"].get(stem)
            if fov_entry:
                st.caption(f"Paired FOV: {fov_entry['name']}")
                st.image(Image.open(io.BytesIO(fov_entry["bytes"])), use_container_width=True)
                if st.button("Remove FOV", key=f"rm_fov_{stem}", use_container_width=True):
                    st.session_state["fov_by_stem"].pop(stem, None)
                    st.rerun()
            else:
                st.caption("No FOV paired.")

            st.divider()
            # Delete image (also drops paired FOV/results)
            if st.button("Delete Image", key=f"del_img_{stem}", use_container_width=True):
                delete_image_by_stem(stem)

    # Always drive the viewer from the current page selection
    st.session_state["selected_stem"] = stem

# ---------------------- Viewer & status ----------------------
st.divider()
viewer = st.container()
with viewer:
    header_cols = st.columns([1.4, 1, 1])
    with header_cols[0]:
        st.markdown("### Viewer")
    with header_cols[1]:
        overlay_toggle = st.toggle("Show overlay", value=True, key="overlay_tog")
    with header_cols[2]:
        alpha = st.slider("Opacity", 0, 100, 50, key="alpha")/100.0 if overlay_toggle else 0.0

    stage = st.empty()  # live stage text “warming up / …”
    img_col, prob_col, out_col = st.columns([1, 1, 1])

    sel_stem = st.session_state.get("selected_stem")
    if sel_stem:
        # find file for selected stem
        img_file = next((f for f in img_files if stem_of(f.name) == sel_stem), None)
        if img_file is None:
            st.warning("Selected image not found.")
        else:
            img = pil_from_upload(img_file)

            # Original + FOV (if any)
            with img_col:
                try_zoomable("Original (zoomable)" if zoomable_image else "Original", img)
                fov_entry = st.session_state["fov_by_stem"].get(sel_stem)
                if fov_entry:
                    st.image(Image.open(io.BytesIO(fov_entry["bytes"])), caption="FOV", use_container_width=True)

            res = st.session_state["results"].get(sel_stem)
            if res:
                # probability map
                with prob_col:
                    prob = res["prob"]
                    st.image(prob, caption="Probability", use_container_width=True, clamp=True)
                # overlay
                with out_col:
                    overlay_rgb = colorize_mask(res["mask"])
                    blended = blend(np.array(img), overlay_rgb, alpha) if overlay_toggle else np.array(img)
                    try_zoomable("Overlay (zoomable)" if zoomable_image else "Overlay", Image.fromarray(blended))
                st.caption(f"Time: {res['timings']['total_ms']:.1f} ms • Device: {res['device']}")

                # Metrics (if GT and computed; GT UI is disabled in this screen)
                if res.get("metrics"):
                    m = res["metrics"]
                    st.markdown(
                        f"**ACC** {m['acc']:.3f} • **SEN** {m['sen']:.3f} • **SPE** {m['spe']:.3f} • "
                        f"**Dice/F1** {m['dice']:.3f} • **IoU** {m['iou']:.3f}"
                    )
            else:
                with prob_col:
                    st.info("Run inference to view probability map.")
                with out_col:
                    st.info("Overlay will appear here after prediction.")

# ---------------------- Inference trigger ----------------------
if btn_run and img_files and (not st.session_state["running"]):
    st.session_state["running"] = True
    st.session_state["stop_flag"] = False
    add_msg("info", "Starting inference run.")
    # Load model (MATFHI for this tab; UNet later for comparison)
    dev = device_label()
    model = load_model("MATFHI", device=dev)

    # loop images
    for f in img_files:
        if st.session_state["stop_flag"]:
            break
        stem = stem_of(f.name)
        im = pil_from_upload(f)
        w, h = im.size

        # Per-image FOV from session mapping
        fov_im = None
        fov_entry = st.session_state["fov_by_stem"].get(stem)
        if fov_entry:
            with contextlib.suppress(Exception):
                fov_im = to_gray(Image.open(io.BytesIO(fov_entry["bytes"]))).resize((w, h), Image.NEAREST)

        # No GT uploader in this UI (leave metrics None unless you re-enable GT)
        gt_im = None

        # Live stages
        stage_runner(stage, "Warming up…")
        time.sleep(0.05)

        t0 = time.time()
        stage_runner(stage, "Preprocessing…")
        # TODO: your real preprocess (resize/normalize/tile)
        time.sleep(0.05)

        if st.session_state["stop_flag"]:
            break
        stage_runner(stage, "Predicting…")
        prob = model.infer(im)  # [H,W] float32 in [0,1]
        time.sleep(0.05)

        if st.session_state["stop_flag"]:
            break
        stage_runner(stage, "Post-processing…")
        mask = (prob >= st.session_state.get("threshold", 0.5) if 'threshold' in st.session_state else prob >= 0.5).astype(np.uint8) * 255
        total_ms = (time.time() - t0) * 1000.0

        # Metrics if GT (none here by default)
        metrics = None
        if gt_im is not None:
            gt_bin  = (np.array(gt_im) > 127).astype(np.uint8)
            pred    = (mask > 127).astype(np.uint8)
            fov_bin = (np.array(fov_im) > 127).astype(np.uint8) if (fov_im is not None and st.session_state.get("submode") == "With Ground Truth" and st.session_state.get("mask_outside_fov", True)) else None
            metrics = compute_basic_metrics(pred, gt_bin, fov_bin)

        st.session_state["results"][stem] = {
            "prob": prob, "mask": mask,
            "timings": {"total_ms": total_ms},
            "device": dev, "metrics": metrics
        }
        st.session_state["selected_stem"] = stem  # focus latest
        stage_runner(stage, "Done.")
        st.rerun()

    st.session_state["running"] = False
    st.session_state["done_once"] = True
    stage_runner(stage, "Idle.")
    st.rerun()

# ---------------------- Messages / Errors bottom ----------------------
st.divider()
st.markdown("#### Messages")
if not st.session_state["messages"]:
    st.caption("No messages yet.")
else:
    for m in st.session_state["messages"]:
        if m["kind"] == "error":
            st.error(m["text"])
        elif m["kind"] == "warn":
            st.warning(m["text"])
        else:
            st.info(m["text"])

# ---------------------- Comparison tab scaffold ----------------------
if st.session_state["mode_top"] == "Comparison (UNet vs MATFHI)":
    st.warning("Comparison mode scaffolded. Later: load UNet + MATFHI and render side-by-side with the same inputs/threshold.")
