# apps/streamlit/app.py
import io, time, pathlib, base64, contextlib
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
import sys 
import cv2

# Model checkpoint file
#MODEL_CHECKPOINT = Path("outputs/checkpoints/[DRIVE] baseunet_dpcn_6_iters_64ch_msu_cbam_hassskip_w_augs_newDataloader_drive_patching.pth")

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.append(str(PROJECT_ROOT))

IMAGE_SIZE = 512
USE_FOV_IN_MODEL = False  # set True ONLY if the checkpoint was trained using model(x, fov=...)

# Model + preprocessing utilities
from src.models.wrappers.dpcn_concat_unet import DPCNConcatUNet
from src.data.preprocessing import _iso_resize_and_pad


DATASET_CHECKPOINTS = {
    "DRIVE":     Path("outputs/checkpoints/[DRIVE] baseunet_dpcn_6_iters_64ch_msu_cbam_hassskip_w_augs_newDataloader_drive_patching.pth"),
    "CHASE-DB1": Path("outputs/checkpoints/[CHASEDB1] baseunet_dpcn_6_iters_64ch_2hl_64rt_cbam16_msu_cbam_hassskip_50_epochs_w_augs_newDataloader_CHASEDB1_patching.pth"),
    "STARE":     Path("outputs/checkpoints/[STARE] baseunet_dpcn_6_iters_64ch_2hl_64rt_cbam16_msu_cbam_hassskip_50_epochs_w_augs_newDataloader_STARE_patching.pth"),
}
IMAGE_SIZE_BY_DATASET = {"DRIVE": 512, "CHASE-DB1": 512, "STARE": 512} 


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
    ss.setdefault("fov_tog", False)
    ss.setdefault("overlay_reset", False)
    ss.setdefault("gt_by_stem", {})  
    ss.setdefault("dataset_choice", "DRIVE")



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

def caption_with_size(label: str, im: Image.Image) -> str:
    w, h = im.size  # PIL: (width, height)
    return f"{label}   |  {w} × {h}px |"

@st.cache_resource
def load_seg_model(device: str = "auto", dataset: Optional[str] = None, ckpt_key: Optional[str] = None):
    dev = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    if device != "auto":
        dev = device

    ds = dataset or st.session_state.get("dataset_choice", "DRIVE")
    ckpt = DATASET_CHECKPOINTS.get(ds)
    if ckpt is None or not ckpt.exists():
        st.error(f"No checkpoint found for '{ds}' at: {ckpt}")
        raise FileNotFoundError(f"Missing checkpoint for {ds}: {ckpt}")

    BASE_KW = {"cbam_reduction": 16}
    model = DPCNConcatUNet(
        in_ch=1, enh_channels=64, iters=6,
        threshold_mode="scaled_vat", half_life=2.0, reduce_to=64,
        base_kwargs=BASE_KW,
    ).to(dev).eval()

    state = torch.load(ckpt, map_location=dev)
    # be robust to different save styles
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    # expose some debug info
    meta = {"dataset": ds, "ckpt_path": str(ckpt)}
    return (model, dev, meta)


def load_fov_1hw_from_bytes(fov_bytes: bytes, target_size: int) -> np.ndarray:
    """
    Decode FOV mask bytes -> grayscale -> isotropic resize+pad -> Otsu -> [1,H,W] float32 in {0,1}.
    Mirrors preprocess_mask but works directly from bytes (no temp files).
    """
    # Decode from bytes (keeps palette/alpha if present)
    buf = np.frombuffer(fov_bytes, dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)  # uint8/uint16, gray/RGB/RGBA
    if m is None:
        raise ValueError("Could not decode FOV bytes")

    # Convert to 8-bit grayscale
    if m.dtype != np.uint8:
        m = cv2.convertScaleAbs(m)  # safe 16-bit -> 8-bit, etc.
    if m.ndim == 3:
        if m.shape[2] == 4:
            m = cv2.cvtColor(m, cv2.COLOR_BGRA2GRAY)
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # Same geometry as training: isotropic resize + pad (nearest for masks)
    m = _iso_resize_and_pad(m, target=target_size, pad_value=0)

    # Hard binary via Otsu, then cast to {0,1} float32 and add channel dim
    m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    m = (m > 0).astype(np.float32)
    return np.expand_dims(m, axis=0).astype(np.float32)  # [1,H,W]


def preprocess_image_retina_from_pil(
    pil_im: Image.Image,
    target_size: int = 512,
    use_gamma: bool = True,
    gamma: float = 0.9,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
) -> np.ndarray:
    """
    Pathless version of preprocess_image_retina:
      - use PIL image directly
      - green channel -> iso resize+pad -> CLAHE -> optional gamma
      - returns (1, H, W) float32 in [0,1]
    """
    # 1) Green channel as uint8
    g_u8 = np.array(pil_im.convert("RGB"), dtype=np.uint8)[..., 1]  # (H,W) uint8

    # 2) Isotropic resize + pad (same function you already import)
    g_u8 = _iso_resize_and_pad(g_u8, target=target_size, pad_value=0)

    # 3) CLAHE in uint8
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    g_eq_u8 = clahe.apply(g_u8)

    # 4) To float32 [0,1]
    g = g_eq_u8.astype(np.float32) / 255.0

    # 5) Optional gamma (np.power on [0,1]; gamma<1 brightens)
    if use_gamma and 0.5 <= gamma <= 1.2:
        g = np.power(g, gamma, dtype=np.float32)

    return np.expand_dims(g, axis=0).astype(np.float32)  # (1,H,W)


def preprocess_mask_from_bytes(mask_bytes: bytes, target_size: int = 512) -> np.ndarray:
    """
    Pathless version of preprocess_mask:
      - bytes -> imdecode -> grayscale -> iso resize+pad (nearest) -> Otsu -> {0,1} float32
      - returns (1,H,W) float32 in {0,1}
    """
    buf = np.frombuffer(mask_bytes, dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError("Could not decode mask bytes")

    if m.dtype != np.uint8:
        m = cv2.convertScaleAbs(m)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGRA2GRAY) if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    m = _iso_resize_and_pad(m, target=target_size, pad_value=0)  # nearest for masks (2D)
    m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    m = (m > 0).astype(np.float32)
    return np.expand_dims(m, axis=0).astype(np.float32)


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

def fov_bin_from_bytes(fov_bytes: bytes, out_wh: tuple[int, int]) -> np.ndarray:
    """Load FOV bytes → grayscale → resize → hard-binarize (0/1)."""
    im = Image.open(io.BytesIO(fov_bytes)).convert("L").resize(out_wh, Image.NEAREST)
    arr = np.array(im, dtype=np.uint8)
    # force binary; >127 becomes inside-FOV (1), else 0
    return (arr > 127).astype(np.uint8)  # [H,W] in {0,1}


def dataset_toggle_row(disabled: bool = False):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.markdown("**Chosen dataset model:**")
    for label, col in [("DRIVE", c2), ("CHASE-DB1", c3), ("STARE", c4)]:
        with col:
            selected = (st.session_state["dataset_choice"] == label)
            if st.button(
                label,
                type=("primary" if selected else "secondary"),
                use_container_width=True,
                disabled=disabled,
            ):
                if not selected:
                    st.session_state["dataset_choice"] = label
                    # Clear any old results since the model will change
                    st.session_state["results"].clear()
                    st.session_state["overlay_reset"] = True
                    st.rerun()


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
st.markdown("# MATHFI: Multi-scale Adaptive Thresholding with Hierarchical Feature Integration")
st.divider()

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
    st.markdown("## Inference")
    sel_stem = st.session_state.get("selected_stem")
    has_result = bool(sel_stem and st.session_state.get("results", {}).get(sel_stem))

    # do we actually have a paired FOV for the selected image?
    fov_entry_hdr = st.session_state.get("fov_by_stem", {}).get(sel_stem)
    has_fov_for_sel = bool(fov_entry_hdr and fov_entry_hdr.get("bytes"))

    header_cols = st.columns([1, 1, 1])
    with header_cols[0]:
        raw_header_cols = st.columns([3, 1])
        with raw_header_cols[0]:
            st.markdown("#### Selected Raw Image")
        with raw_header_cols[1]:
            # Only render the toggle if a FOV is paired; otherwise clear stale state
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
            
            # --- Viewer: Original + FOV overlay ---
            with img_col:
                # Start from the raw uploaded image (RGB)
                base_pil = img  
                W0, H0 = base_pil.size

                # If FOV exists and toggle is ON: black out pixels outside the FOV
                fov_entry = st.session_state.get("fov_by_stem", {}).get(sel_stem)
                if fov_entry and st.session_state.get("fov_tog", False):
                    fov01 = fov_bin_from_bytes(fov_entry["bytes"], (W0, H0))  # [H0,W0] in {0,1}
                    base_rgb = np.array(base_pil, dtype=np.uint8)             # [H0,W0,3]
                    # keep vessels/retina as-is inside FOV; set outside to black
                    base_rgb[fov01 == 0] = 0
                    show_pil = Image.fromarray(base_rgb)
                    try_zoomable(caption_with_size("Original (FOV applied)", show_pil), show_pil)
                else:
                    try_zoomable(caption_with_size("Original (Raw image input)", base_pil), base_pil)


           

            # Results panes if exist
            res = st.session_state["results"].get(sel_stem)
            if res:
                # Middle column = Preprocessed Image
                with prob_col:
                    pre_img = res.get("pre")
                    if pre_img is not None:
                        st.image(pre_img, caption="Preprocessed Image | 512 × 512px |", use_container_width=True, clamp=True)
                        st.markdown("#")
                    else:
                        st.warning("No preprocessed image saved.")
                with out_col:
                    # Optional overlay on the resized+pad RGB (same 512×512 geometry)
                    overlay_on = has_result and st.session_state.get("overlay_tog", False)

                    if overlay_on:
                        # live threshold → recompute binary map from probs
                        thr = st.session_state.get("threshold", 0.5)
                        prob_np = res["probs"]  # [H,W] float32 (FOV-clamped)
                        mask_bin = (prob_np >= thr).astype(np.uint8) * 255  # [H,W] 0/255

                        alpha_pct = st.session_state.get("alpha", 50)
                        alpha = alpha_pct / 100.0

                        base_rgb = res.get("overlay_base_rgb")  # [H,W,3] uint8 @ model size
                        if base_rgb is None:
                            base_rgb = _iso_resize_and_pad(
                                np.array(img.convert("RGB")), target=IMAGE_SIZE, pad_value=0
                            ).astype(np.uint8)

                        # Per-pixel alpha only where vessels are present (white overlay, transparent elsewhere)
                        base = base_rgb.astype(np.float32)
                        mask01 = (mask_bin.astype(np.float32) / 255.0)[..., None]  # [H,W,1]
                        alpha_map = alpha * mask01                                  # [H,W,1]
                        blended = np.clip(base * (1.0 - alpha_map) + 255.0 * alpha_map, 0, 255).astype(np.uint8)

                        try_zoomable("Overlay (zoomable)" if zoomable_image else "Overlay", Image.fromarray(blended))
                        st.slider("Opacity", 0, 100, alpha_pct, key="alpha")
                    
                    else:
                        prob_np = res["probs"]  # [H,W] float32 (already FOV-clamped)
                        thr = st.session_state.get("threshold", 0.5)
                        mask_bin = (prob_np >= thr).astype(np.uint8) * 255  # [H,W] 0/255

                        # Show the binary map
                        st.image(mask_bin, caption="Predicted Vessel Map (binary)", use_container_width=True)





                    
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
                        pre_img = res.get("pre")
                        if pre_img is not None:
                            st.image(pre_img, caption="Preprocessed Image (FOV-applied)", use_container_width=True, clamp=True)
                            st.markdown("#")
                        else:
                            st.warning("No preprocessed image saved.")

                    with out_col:
                        st.image(ph_rgb, caption="Predicted Vessel Map (placeholder)", use_container_width=True)
                        
                else:
                    with prob_col:
                        st.warning("⚠️ Run inference to view preprocessed image.")
                    with out_col:
                        st.warning("⚠️ Run inference to view probability map.")
                        

                    
    else:
        st.info("No image selected. Choose one in the Selection gallery above.")
    
    run_columns_viewer = st.columns([1,1,1])

    dataset_toggle_row(disabled=st.session_state.get("running", False))

    with run_columns_viewer[0]:
        btn_run_viewer = st.button(
            "Run Inference",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("running", False) or not has_selected_file
        )

    with run_columns_viewer[1]:
        btn_stop_viewer = st.button(
            "Stop",
            use_container_width=True,
            disabled=(not st.session_state.get("running", False))
        )

    with run_columns_viewer[2]:
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

        # Load model (cached)
        ds = st.session_state.get("dataset_choice", "DRIVE")
        IMAGE_SIZE = IMAGE_SIZE_BY_DATASET.get(ds, 512)

        model, dev, meta = load_seg_model(dataset=ds, ckpt_key=str(DATASET_CHECKPOINTS[ds]))

        # 1) Preprocess (grayscale + retina preprocessing)
        stage_runner(stage, "Preprocessing…"); time.sleep(0.05)
        fundus_pil = pil_from_upload(img_file)
        img_1hw = preprocess_image_retina_from_pil(
            fundus_pil,
            target_size=IMAGE_SIZE,
            use_gamma=True,
            gamma=0.9,
            clahe_clip=2.0,
            clahe_tiles=8,
        ).astype(np.float32)  # (1,H,W) in [0,1]


        H, W = img_1hw.shape[-2], img_1hw.shape[-1]

        fov_entry = st.session_state["fov_by_stem"].get(sel_stem)
        if fov_entry and fov_entry.get("bytes"):
            fov_1hw = load_fov_1hw_from_bytes(fov_entry["bytes"], target_size=IMAGE_SIZE)
        else:
            # fallback: use nonzero (pads stay 0) from preprocessed image
            fov_1hw = (img_1hw > 0).astype(np.float32)


        # --- Gate the image by FOV BEFORE the model (as in dataset.py) ---
        img_fov_1hw = (img_1hw * fov_1hw).astype(np.float32)

        # --- UI preprocessed preview (FOV-applied) ---
        pre_img_vis = (img_fov_1hw[0] * 255.0).astype(np.uint8)  # [H,W] uint8

        # --- Tensors ---
        x   = torch.from_numpy(img_fov_1hw).unsqueeze(0).to(dev)  # [1,1,H,W]
        fov = torch.from_numpy(fov_1hw).unsqueeze(0).to(dev)      # [1,1,H,W]

        # 4) Inference
        stage_runner(stage, "Predicting…"); time.sleep(0.05)
        t0 = time.time()
        with torch.no_grad():
            use_amp = (dev == "cuda")
            amp_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(x, fov=fov) if USE_FOV_IN_MODEL else model(x)
        probs = torch.sigmoid(logits)  # [1,1,H,W]
        total_ms = (time.time() - t0) * 1000.0


        # 5) Threshold and convert for UI
        probs = probs * (fov > 0.5).float()
        thr = st.session_state.get("threshold", 0.5)
        pred01 = (probs >= thr).float()

        mask_bin_u8 = (pred01[0,0].cpu().numpy() * 255).astype(np.uint8)   # [H,W] 0/255
        prob_np     =  probs[0,0].cpu().numpy().astype(np.float32)         # [H,W] 0..1


        # 6) (Optional) Metrics — only if “With Ground Truth” and GT exists for this stem
        metrics = None
        if st.session_state.get("submode") == "With Ground Truth":
            gt_entry = st.session_state.get("gt_by_stem", {}).get(sel_stem)
            if gt_entry and gt_entry.get("bytes"):
                # preprocess GT to same size
                gt_1hw = preprocess_mask_from_bytes(gt_entry["bytes"], target_size=IMAGE_SIZE).astype(np.float32)
                y = torch.from_numpy(gt_1hw).unsqueeze(0).to(dev)      # [1,1,H,W]
                m = (fov > 0.5).float()                                # FOV mask
                # quick scores (we can swap in this with package’s iou/dice/etc.)
                tp = (pred01*m*y).sum().item()
                fn = ((1-pred01)*m*y).sum().item()
                fp = (pred01*m*(1-y)).sum().item()
                dice = (2*tp) / max(1.0, 2*tp + fp + fn)
                iou  = tp / max(1.0, tp + fp + fn)
                metrics = {"dice": float(dice), "iou": float(iou)}

        # RGB base for overlay (match model space: 512×512, isotropic pad)
        overlay_base_rgb = _iso_resize_and_pad(
            np.array(fundus_pil.convert("RGB")), target=IMAGE_SIZE, pad_value=0
        ).astype(np.uint8)  # [H,W,3] uint8


        # 7) Save to session for the Viewer
        st.session_state["results"][sel_stem] = {
            "probs": prob_np,           # float32 [H,W] 0..1 (keep if we need it later)
            "mask":  mask_bin_u8,          # uint8 [H,W] 0/255
            "pre":   pre_img_vis,           # uint8 [H,W]
            "overlay_base_rgb": overlay_base_rgb,
            "timings": {"total_ms": total_ms},
            "device": dev,
            "metrics": metrics
        }
        stage_runner(stage, "Done.")
        st.session_state["running"] = False
        st.session_state["done_once"] = True
        st.rerun()



# ---------------------- Comparison tab scaffold ----------------------
if st.session_state["mode_top"] == "Comparison (UNet vs MATFHI)":
    st.warning("Comparison mode scaffolded. Later: load UNet + MATFHI and render side-by-side with the same inputs/threshold.")
