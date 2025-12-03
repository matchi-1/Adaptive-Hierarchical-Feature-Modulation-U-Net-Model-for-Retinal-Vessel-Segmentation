from typing import Optional, Tuple, Dict, Any
import streamlit as st
import torch
from src.models.wrappers.dpcn_concat_unet import DPCNConcatUNet
from src.models.orig_unet import UNet 
from .config import DATASET_CHECKPOINTS, UNET_CHECKPOINTS
from pathlib import Path

@st.cache_resource(show_spinner=False)
def load_mathfi_model(device: str = "auto", dataset: Optional[str] = None):
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
        base_kwargs={"cbam_reduction": 16,},
        refine_edge=True,
    ).to(dev).eval()

    state = torch.load(ckpt, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model, dev, {"dataset": ds, "ckpt_path": str(ckpt)}




def load_unet_model(*, dataset: str, checkpoints: dict[str, Path], device: str = "auto"):
    dev = "cuda" if (torch.cuda.is_available()) else "cpu"
    if device != "auto":
        dev = device

    ckpt = checkpoints.get(dataset)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"UNet checkpoint not found for '{dataset}' at: {ckpt}")

    model = UNet(in_channels=1, out_channels=2).to(dev).eval()
    state = torch.load(ckpt, map_location=dev)

    # be robust to different save styles
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    meta = {"dataset": dataset, "ckpt_path": str(ckpt)}
    return (model, dev, meta)