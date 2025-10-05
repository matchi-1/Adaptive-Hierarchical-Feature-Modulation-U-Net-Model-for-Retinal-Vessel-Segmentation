# apps/streamlit/app.py
import io, time, pathlib, base64, contextlib
from typing import Dict, Any, Optional, List

import numpy as np
from PIL import Image
import streamlit as st

# Optional deps (safe fallbacks if missing)
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

# ---------------------- Page & State ----------------------
st.set_page_config(page_title="Retinal Vessel Segmentation UI", layout="wide")

def init_state():
    ss = st.session_state
    ss.setdefault("mode_top", "Single Model (MATFHI)")
    ss.setdefault("submode", "Predict Only")  # or "With Ground Truth"
    ss.setdefault("selected_stem", None)
    ss.setdefault("files_img", [])   # uploaded fundus files (right side)
    ss.setdefault("files_gt", [])    # optional GT files
    ss.setdefault("files_fov", [])   # optional FOV files
    ss.setdefault("results", {})     # stem -> dict(prob, mask, timings, metrics, messages)
    ss.setdefault("messages", [])    # bottom console messages
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

def map_by_stem(files) -> Dict[str, Any]:
    return {stem_of(f.name): f for f in (files or [])}

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
    # TODO: replace with real model loading (MATFHI vs UNet)
    class Dummy:
        name = model_name
        def infer(self, img: Image.Image) -> np.ndarray:
            # Return a fake probability map [H,W] in [0,1] just to prove UI
            w, h = img.size
            yy, xx = np.mgrid[0:h, 0:w]
            prob = (np.sin(xx / 20.0) * np.cos(yy / 25.0) + 1.0) * 0.5
            # Add a soft circular vessel-ish hint
            cx, cy = w // 2, h // 2
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            prob += np.exp(-(r / (min(w, h) / 3.0)) ** 2) * 0.25
            return np.clip(prob, 0, 1).astype(np.float32)
    return Dummy()

def render_telemetry():
    dev = device_label()
    with st.container(border=True):
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
            st.write(f"GPU: none (using {dev})")

def try_zoomable(label: str, img: Image.Image):
    if zoomable_image:
        # Zoomable PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        zoomable_image(f"data:image/png;base64,{b64}", label=label, height=520)
    else:
        st.image(img, caption=label, use_container_width=True)

def stage_runner(stage_placeholder, text):
    stage_placeholder.markdown(f"<div style='padding:4px;background:#111;border-radius:6px;color:#eee'>"
                               f"<b>Status:</b> {text}</div>", unsafe_allow_html=True)

def clear_session_outputs():
    st.session_state["results"] = {}
    st.session_state["messages"] = []
    st.session_state["selected_stem"] = None
    st.session_state["running"] = False
    st.session_state["stop_flag"] = False
    st.session_state["done_once"] = False

# ---------------------- Top Controls ----------------------
model_mode = st.sidebar.selectbox("Top Mode", ["Single Model (MATFHI)", "Comparison (UNet vs MATFHI)"],
                                  index=0, key="mode_top")
submode = st.sidebar.radio("Run Mode", ["Predict Only", "With Ground Truth"], key="submode")
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
mask_outside_fov = st.sidebar.checkbox("Mask outside FOV (metrics)", value=True)
st.sidebar.divider()
btn_run = st.sidebar.button("Run Inference", type="primary", use_container_width=True)
btn_stop = st.sidebar.button("Stop", use_container_width=True, disabled=st.session_state["running"] is False)
# Reset appears after a run completes
if st.session_state["done_once"]:
    btn_reset = st.sidebar.button("Reset", use_container_width=True)
else:
    btn_reset = False

if btn_reset:
    clear_session_outputs()
    st.rerun()

if btn_stop:
    st.session_state["stop_flag"] = True
    add_msg("info", "Stop requested; finishing current step…")

# ---------------------- Layout (Left info | Right viewer) ----------------------
left, right = st.columns([1.1, 1.9])

# Left: telemetry & notes
with left:
    render_telemetry()
    st.markdown("### Notes")
    st.write("- Upload images on the **right**. Click thumbnails to view.")
    st.write("- Toggle **Overlay** to blend mask over original (default 50%).")
    st.write("- Metrics appear in **With Ground Truth** mode.")

# Right: uploader & viewer 
with right:
    # Uploader row (right side)
    st.markdown("#### Inputs")
    up1 = st.file_uploader("Fundus image(s)", type=["png","jpg","jpeg","tif"], accept_multiple_files=True, key="u1")
    up2 = st.file_uploader("FOV mask(s) (optional)", type=["png","jpg","jpeg","tif"], accept_multiple_files=True, key="u2")
    up3 = None
    if submode == "With Ground Truth":
        up3 = st.file_uploader("Ground truth mask(s)", type=["png","jpg","jpeg","tif"], accept_multiple_files=True, key="u3")

    # Keep uploaded files in session (so Reset can clear them)
    if up1 is not None:
        st.session_state["files_img"] = up1
    if up2 is not None:
        st.session_state["files_fov"] = up2
    if up3 is not None:
        st.session_state["files_gt"] = up3

    img_files = st.session_state["files_img"]
    fov_map = map_by_stem(st.session_state["files_fov"])
    gt_map  = map_by_stem(st.session_state["files_gt"]) if submode == "With Ground Truth" else {}

    # Thumbnail strip
    if img_files:
        st.markdown("#### Selection")
        cols = st.columns(min(6, len(img_files)))
        for i, f in enumerate(img_files):
            thumb = Image.open(f).convert("RGB")
            thumb_small = thumb.copy()
            thumb_small.thumbnail((180, 180))
            with cols[i % len(cols)]:
                if st.button(stem_of(f.name), key=f"pick_{i}"):
                    st.session_state["selected_stem"] = stem_of(f.name)
                st.image(thumb_small, use_container_width=True)

    # Pick current
    sel_stem = st.session_state["selected_stem"]
    if not sel_stem and img_files:
        sel_stem = stem_of(img_files[0].name)
        st.session_state["selected_stem"] = sel_stem

    # Viewer & status
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

        if sel_stem:
            # retrieve handles
            img_file = next((f for f in img_files if stem_of(f.name) == sel_stem), None)
            if img_file is None:
                st.warning("Selected image not found.")
            else:
                img = pil_from_upload(img_file)
                # Show the three panes; populate after run if results exist
                with img_col:
                    try_zoomable("Original (zoomable)" if zoomable_image else "Original", img)

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

                    # Metrics (if GT and computed)
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
                if st.session_state["stop_flag"]: break
                stem = stem_of(f.name)
                im = pil_from_upload(f)
                w, h = im.size

                # FOV/GT matching
                fov_im = None
                if stem in fov_map:
                    with contextlib.suppress(Exception):
                        fov_im = to_gray(Image.open(fov_map[stem])).resize((w, h), Image.NEAREST)
                gt_im = None
                if submode == "With Ground Truth" and stem in gt_map:
                    with contextlib.suppress(Exception):
                        gt_im = to_gray(Image.open(gt_map[stem])).resize((w, h), Image.NEAREST)

                # Live stages
                stage_runner(stage, "Warming up…")
                time.sleep(0.05)  # tiny pause for UX

                t0 = time.time()
                stage_runner(stage, "Preprocessing…")
                # TODO: preprocess (resize/normalize/tile)
                time.sleep(0.05)

                if st.session_state["stop_flag"]: break
                stage_runner(stage, "Predicting…")
                prob = model.infer(im)  # [H,W] float32 in [0,1]
                time.sleep(0.05)

                if st.session_state["stop_flag"]: break
                stage_runner(stage, "Post-processing…")
                mask = (prob >= threshold).astype(np.uint8) * 255
                total_ms = (time.time() - t0) * 1000.0

                # Metrics if GT
                metrics = None
                if gt_im is not None:
                    gt_bin  = (np.array(gt_im) > 127).astype(np.uint8)
                    pred    = (mask > 127).astype(np.uint8)
                    fov_bin = (np.array(fov_im) > 127).astype(np.uint8) if (fov_im is not None and mask_outside_fov) else None
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
if model_mode == "Comparison (UNet vs MATFHI)":
    st.warning("Comparison mode scaffolded. Later: load UNet + MATFHI and render side-by-side with the same inputs/threshold.")
