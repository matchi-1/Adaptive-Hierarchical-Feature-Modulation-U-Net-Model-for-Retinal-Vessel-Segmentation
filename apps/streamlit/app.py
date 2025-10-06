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
    ss.setdefault("deleted_stems", set())              # <- NEW: keep stems that were deleted
    ss.setdefault("uploader_nonce", 0)                 # <- NEW: bump to reset uploader widget
    ss.setdefault("fov_uploader_nonce", {})   # per-stem nonce to reset FOV uploader
    ss.setdefault("overlay_tog", False)
    ss.setdefault("overlay_reset", False)


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
    card_telemetry = st.container(border=True)
    with card_telemetry:
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
    """Remove image (and paired FOV/results) by stem, reset uploader, then rerun."""
    # Remove from our library
    st.session_state["files_img"] = [
        f for f in st.session_state.get("files_img", []) if stem_of(f.name) != stem
    ]
    # Remember deletion so future uploads from the widget don't re-add it
    st.session_state["deleted_stems"].add(stem)
    # Remove paired FOV and results
    st.session_state["fov_by_stem"].pop(stem, None)
    st.session_state["results"].pop(stem, None)
    # Fix selection/pagination
    if st.session_state.get("selected_stem") == stem:
        st.session_state["selected_stem"] = None
    n = len(st.session_state.get("files_img", []))
    st.session_state["sel_idx"] = 0 if n == 0 else min(st.session_state["sel_idx"], n - 1)
    # Reset uploader widget so its visual list clears
    st.session_state["uploader_nonce"] += 1
    st.session_state["fov_uploader_nonce"].pop(stem, None)  # NEW: reset uploader for this stem
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
    st.session_state["deleted_stems"] = set()
    st.session_state["uploader_nonce"] += 1  # also reset uploader
    st.session_state["fov_uploader_nonce"] = {}  # reset fovs

# ---------------------- Sidebar (top controls + sticky footer) ----------------------
top = st.sidebar.container()
footer = st.sidebar.container()  # styled to bottom via CSS

with top:
    model_mode = st.selectbox("Top Mode", ["Single Model (MATFHI)", "Comparison (UNet vs MATFHI)"],
                              index=0, key="mode_top")
    submode = st.radio("Run Mode", ["Predict Only", "With Ground Truth"], key="submode")
    

with footer:
    st.markdown("#")
    render_telemetry_sidebar_footer()
    st.markdown("#")
    st.markdown("### How to use")
    st.write("- Upload a **batch of fundus images** below.")
    st.write("- Use **Prev/Next** to browse; upload a **per-image FOV** on the right.")
    st.write("- FOV is automatically **paired** with its image.")
    st.write("- Run inference from the sidebar; overlay/timing show in the Viewer.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Main layout ----------------------
st.markdown("####")
card_upload_sec = st.container(border=True)
with card_upload_sec:
    st.markdown("## Upload Raw Fundus Image(s) here")
    # Key includes nonce so deleting an image resets/clears the widget selection UI
    up1 = st.file_uploader(
        "Fundus images (batch upload)",
        type=["png","jpg","jpeg","tif"],
        accept_multiple_files=True,
        key=f"u1_{st.session_state['uploader_nonce']}"
    )

# Merge new uploads into our library, ignoring stems that were deleted
# --- Sync uploader content to library exactly (handles deletions in uploader UI) ---
if up1 is not None:
    # Current library (before sync)
    old_map = {stem_of(f.name): f for f in st.session_state.get("files_img", [])}

    # New library = exactly what's listed in the uploader (minus stems deleted from Selection)
    new_items: list = []
    new_stems: list = []
    skip_stems = st.session_state.get("deleted_stems", set())  # stems deleted via Selection

    for f in up1:
        s = stem_of(f.name)
        if s in skip_stems:
            # If the user deleted it from Selection, ignore it even if still shown by the uploader
            continue
        new_items.append(f)
        new_stems.append(s)

    # Anything that existed but is no longer in the uploader list is truly removed
    removed_stems = set(old_map.keys()) - set(new_stems)
    for s in removed_stems:
        st.session_state["fov_by_stem"].pop(s, None)
        st.session_state["results"].pop(s, None)
        if st.session_state.get("selected_stem") == s:
            st.session_state["selected_stem"] = None

    # Commit the exact list, and keep pagination/index valid
    st.session_state["files_img"] = new_items
    n = len(new_items)
    st.session_state["sel_idx"] = 0 if n == 0 else min(st.session_state.get("sel_idx", 0), n - 1)


img_files = st.session_state["files_img"]



# ---------------------- Selection (Prev/Next with explicit Select/Unselect) ----------------------
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
            st.session_state["sel_idx"] = max(0, idx - 1)
            st.rerun()
    with c_mid:
        st.write(f"Image {idx+1}/{n}")
    with c_next:
        if st.button("Next ▶", use_container_width=True, disabled=(idx >= n - 1)):
            st.session_state["sel_idx"] = min(n - 1, idx + 1)
            st.rerun()

    # Current item card
    stem = stems[idx]
    file_obj = img_files[idx]
    img = Image.open(file_obj).convert("RGB")
    is_selected = (st.session_state.get("selected_stem") == stem)

    
    card = st.container(border=True)
    with card:
        top_cols = st.columns([2, 1])

        # LEFT: name, big selected banner, image, select/unselect button
        with top_cols[0]:
            st.markdown(f"**{file_obj.name}**")

            st.image(img, use_container_width=True)


        # RIGHT: per-image FOV controls and delete
        with top_cols[1]:
            # Big selected / not selected banner
            banner_txt = "✅ SELECTED" if is_selected else "NOT SELECTED"
            banner_col = "#0da2a2" if is_selected else "#c45959"
            st.markdown(
                f"<div style='font-size:1.35rem;font-weight:500;color:{banner_col};"
                f"margin:.25rem 0 .5rem 0'>{banner_txt}</div>",
                unsafe_allow_html=True
            )

            # ---- FOV uploader with per-stem nonce (keeps uploader & Remove in sync) ----
            nonce = st.session_state["fov_uploader_nonce"].get(stem, 0)
            fov_up = st.file_uploader(
                f"FOV for {stem}",
                type=["png","jpg","jpeg","tif"],
                key=f"fov_sel_{stem}_{nonce}"     # KEY INCLUDES NONCE
            )
            if fov_up is not None:
                st.session_state["fov_by_stem"][stem] = {
                    "name": fov_up.name,
                    "mime": fov_up.type or "image/png",
                    "bytes": fov_up.getvalue(),
                }
                # bump nonce to clear uploader’s selected file display
                st.session_state["fov_uploader_nonce"][stem] = nonce + 1
                st.rerun()

            # Show / remove paired FOV
            fov_entry = st.session_state["fov_by_stem"].get(stem)
            if fov_entry:
                st.image(Image.open(io.BytesIO(fov_entry["bytes"])), use_container_width=True)
                if st.button("Remove FOV", key=f"rm_fov_sel_{stem}", use_container_width=True):
                    st.session_state["fov_by_stem"].pop(stem, None)
                    # bump nonce again so uploader clears after removal
                    st.session_state["fov_uploader_nonce"][stem] = st.session_state["fov_uploader_nonce"].get(stem, 0) + 1
                    st.rerun()
            else:
                st.caption("No FOV paired.")


            #st.divider()
            # Select / Unselect buttons
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            if st.button("Select this image", key=f"select_{stem}", use_container_width=True,
                            disabled=is_selected):
                st.session_state["selected_stem"] = stem
                st.rerun()
        with c_sel2:
            if st.button("Unselect", key=f"unselect_{stem}", use_container_width=True,
                            disabled=not is_selected):
                st.session_state["selected_stem"] = None
                st.rerun()

        # Delete full image button
        if st.button("Delete Image / Image Pair", key=f"del_img_{stem}", use_container_width=True):
            delete_image_by_stem(stem)
    

# ---------------------- Viewer & status (Run/Stop moved here) ----------------------
st.divider()
viewer = st.container()
with viewer:
    
    sel_stem = st.session_state.get("selected_stem")
    has_result = bool(sel_stem and st.session_state.get("results", {}).get(sel_stem))

    # make sure the selected stem actually exists in the current file list
    has_selected_file = bool(
        sel_stem and any(stem_of(f.name) == sel_stem for f in img_files)
    )

        
    # --- compute selection/result state BEFORE building header controls ---
    sel_stem = st.session_state.get("selected_stem")
    has_result = bool(sel_stem and st.session_state.get("results", {}).get(sel_stem))
    header_cols = st.columns([1, 1, 1])
    with header_cols[0]:
        st.markdown("#### Selected Raw Image")
    with header_cols[1]:
        st.markdown("#### Preprocessed Image")
    with header_cols[2]:
        predicted_header_vessel_map_cols = st.columns([2, 1])
        with predicted_header_vessel_map_cols[0]:
            st.markdown("#### Predicted Vessel Map")
        with predicted_header_vessel_map_cols[1]:
            
            if st.session_state.pop("overlay_reset", False):
                # reset the toggle & alpha on the next render, before widget creation
                st.session_state.pop("overlay_tog", None)  # remove so toggle starts fresh (False)
                st.session_state.pop("alpha", None)

            overlay_toggle = st.toggle("Overlay", key="overlay_tog", disabled=not has_result)
        
    
    stage = st.empty()  # live stage text “warming up / …”
    img_col, prob_col, out_col = st.columns([1, 1, 1])

    #sel_stem = st.session_state.get("selected_stem")
    if sel_stem:
        # Find file for selected stem
        img_file = next((f for f in img_files if stem_of(f.name) == sel_stem), None)
        if img_file is None:
            st.warning("Selected image not found.")
        else:
            img = pil_from_upload(img_file)
            
            # --- Viewer: Original + FOV (display-only) ---
            with img_col:
                try_zoomable("Original (zoomable)" if zoomable_image else "Original", img)

                fov_entry = st.session_state["fov_by_stem"].get(sel_stem)

                with st.expander(f"FOV for {sel_stem}", expanded=False):
                    if fov_entry:
                        # show just the FOV image (no "paired" caption)
                        st.image(Image.open(io.BytesIO(fov_entry["bytes"])), use_container_width=True)
                    else:
                        st.caption("No FOV paired.")

           

            # Results panes if exist
            res = st.session_state["results"].get(sel_stem)
            if res:
                # Middle column = Preprocessed Image
                with prob_col:
                    pre_img = res.get("pre")
                    if pre_img is not None:
                        st.image(pre_img, caption="Preprocessed Image", use_container_width=True, clamp=True)
                        st.markdown("#")
                    else:
                        st.warning("No preprocessed image saved.")
                with out_col:
                    overlay_on = has_result and st.session_state.get("overlay_tog", False)

                    # Use last slider value if present, else default to 50
                    alpha_pct = st.session_state.get("alpha", 50)
                    alpha = (alpha_pct / 100.0) if overlay_on else 0.0

                    overlay_rgb = colorize_mask(res["mask"])
                    blended = blend(np.array(img), overlay_rgb, alpha) if overlay_on else np.array(img)

                    # 1) Show the image first
                    try_zoomable("Overlay (zoomable)" if zoomable_image else "Predicted Vessel Map (w/ Overlay)", Image.fromarray(blended)) # this shouldnt have (w/ overlay) if its still disabled

                    # 2) Only then show the slider — and only if overlay is actually ON
                    if overlay_on:
                        st.slider("Opacity", 0, 100, alpha_pct, key="alpha")


                    
                st.caption(f"Time: {res['timings']['total_ms']:.1f} ms • Device: {res['device']}")
                if res.get("metrics"):
                    m = res["metrics"]
                    st.markdown(
                        f"**ACC** {m['acc']:.3f} • **SEN** {m['sen']:.3f} • **SPE** {m['spe']:.3f} • "
                        f"**Dice/F1** {m['dice']:.3f} • **IoU** {m['iou']:.3f}"
                    )
            else:
                if st.session_state.get("running"):
                    # Placeholders at the same size as the source image
                    w, h = img.size
                    ph_gray = Image.new("L", (w, h), 128)          # mid-gray
                    ph_rgb  = Image.new("RGB", (w, h), (48, 48, 48))  # dark gray

                    with prob_col:
                        st.image(ph_gray, caption="Preprocessed (placeholder)", use_container_width=True)
                    with out_col:
                        st.image(ph_rgb, caption="Predicted Vessel Map (placeholder)", use_container_width=True)
                        
                else:
                    with prob_col:
                        st.warning("⚠️ Run inference to view preprocessed image.")
                    with out_col:
                        st.warning("⚠️ Run inference to view probability map.")
                        

                    
    else:
        st.info("No image selected. Choose one in the Selection gallery above.")

    
    btn_run_viewer = st.button(
        "Run Inference",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("running", False) or not has_selected_file
    )

    btn_stop_viewer = st.button(
        "Stop",
        use_container_width=True,
        disabled=(not st.session_state.get("running", False))
    )

    btn_clear_viewer = st.button(
        "Clear",
        use_container_width=True,
        disabled=not has_result
    )


    # Clear only the preprocessed & prediction for the selected image
    if btn_clear_viewer and sel_stem:
        st.session_state["results"].pop(sel_stem, None)
        st.session_state["overlay_reset"] = True
        st.rerun()

    # Allow stopping from viewer
    if btn_stop_viewer:
        st.session_state["stop_flag"] = True
        add_msg("info", "Stop requested; finishing current step…")

# ---------------------- Inference trigger (viewer-scoped; runs on selected only) ----------------------
if btn_run_viewer and has_selected_file and (not st.session_state.get("running", False)):
    sel_stem = st.session_state["selected_stem"]
    img_file = next((f for f in st.session_state["files_img"] if stem_of(f.name) == sel_stem), None)
    if img_file is None:
        add_msg("error", "Selected image not found.")
    else:
        st.session_state["running"] = True
        st.session_state["stop_flag"] = False
        try:
            add_msg("info", f"Starting inference for **{img_file.name}**.")
            dev = device_label()
            model = load_model("MATFHI", device=dev)
            thr = st.session_state.get("threshold", 0.5)

            im = pil_from_upload(img_file)
            w, h = im.size

            fov_im = None
            fov_entry = st.session_state["fov_by_stem"].get(sel_stem)
            if fov_entry:
                with contextlib.suppress(Exception):
                    fov_im = to_gray(Image.open(io.BytesIO(fov_entry["bytes"]))).resize((w, h), Image.NEAREST)

            stage_runner(stage, "Warming up…"); time.sleep(0.05)
            t0 = time.time()
            stage_runner(stage, "Preprocessing…"); time.sleep(0.05)
            if st.session_state["stop_flag"]:
                stage_runner(stage, "Stopped.")
                pass  

            stage_runner(stage, "Predicting…")
            prob = model.infer(im); time.sleep(0.05)

            stage_runner(stage, "Post-processing…")
            mask = (prob >= thr).astype(np.uint8) * 255
            pre_img = np.array(to_gray(im))
            total_ms = (time.time() - t0) * 1000.0

            st.session_state["results"][sel_stem] = {
                "prob": prob,
                "mask": mask,
                "pre": pre_img,
                "timings": {"total_ms": total_ms},
                "device": dev,
                "metrics": None,
            }
            stage_runner(stage, "Done.")
        finally:
            st.session_state["running"] = False
            st.session_state["done_once"] = True
            st.rerun()


# ---------------------- Comparison tab scaffold ----------------------
if st.session_state["mode_top"] == "Comparison (UNet vs MATFHI)":
    st.warning("Comparison mode scaffolded. Later: load UNet + MATFHI and render side-by-side with the same inputs/threshold.")
