import base64, io
from PIL import Image
import streamlit as st
import numpy as np
from typing import Optional
try:
    from streamlit_zoomable_image import zoomable_image as _zoom
except Exception:
    _zoom = None

def caption_with_size(label: str, im: Image.Image) -> str:
    w, h = im.size
    return f"{label}   |  {w} × {h}px |"

def try_zoomable(label: str, img: Image.Image):
    if _zoom:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        _zoom(f"data:image/png;base64,{b64}", label=label, height=520)
    else:
        st.image(img, caption=label, use_container_width=True)

def stage_runner(stage_placeholder, text: str):
    stage_placeholder.markdown(
        f"<div class='status-ribbon'><b>Status:</b> {text}</div>",
        unsafe_allow_html=True
    )

def render_telemetry_sidebar_footer(psutil=None, torch=None):
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
            st.write("GPU: none (using cpu)")

def dataset_toggle_row(disabled: bool = False):
    from .config import DATASET_CHECKPOINTS
    c1, *cols = st.columns([2] + [1] * len(DATASET_CHECKPOINTS))
    with c1:
        st.markdown("**Chosen dataset model:**")
    for label, col in zip(DATASET_CHECKPOINTS.keys(), cols):
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
                    st.session_state["results"].clear()
                    st.session_state["overlay_reset"] = True
                    st.rerun()
