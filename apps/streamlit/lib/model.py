from typing import Optional, Tuple, Dict, Any
import streamlit as st
import torch
from src.models.wrappers.dpcn_concat_unet import DPCNConcatUNet
from .config import DATASET_CHECKPOINTS

@st.cache_resource(show_spinner=False)
def load_seg_model(device: str = "auto", dataset: Optional[str] = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "auto":
        dev = device

    ds = dataset or "DRIVE"
    ckpt = DATASET_CHECKPOINTS.get(ds)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for {ds}: {ckpt}")

    model = DPCNConcatUNet(
        in_ch=1, enh_channels=64, iters=6,
        threshold_mode="scaled_vat", half_life=2.0, reduce_to=64,
        base_kwargs={"cbam_reduction": 16},
    ).to(dev).eval()

    state = torch.load(ckpt, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model, dev, {"dataset": ds, "ckpt_path": str(ckpt)}
