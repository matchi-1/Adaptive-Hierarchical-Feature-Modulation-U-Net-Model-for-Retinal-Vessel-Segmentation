import torch
from typing import Tuple
from src.models.wrappers.dpcn_concat_unet import DPCNConcatUNet

def build_dpcn_model(device: str | None = None) -> Tuple[torch.nn.Module, str]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DPCNConcatUNet(
        in_ch=1, enh_channels=64, iters=6,
        threshold_mode="scaled_vat", half_life=2.0, reduce_to=64,
        base_kwargs={"cbam_reduction": 16}, refine_edge=True
    ).to(dev).eval()
    return model, dev

def _strip_prefix_if_present(state_dict: dict, prefix: str = "module.") -> dict:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def load_weights_into_model(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
        return
    except Exception:
        pass
    state2 = _strip_prefix_if_present(state, "module.")
    model.load_state_dict(state2, strict=False)

def load_dpcn_from_ckpt(ckpt_path: str, device: str | None = None) -> Tuple[torch.nn.Module, str]:
    model, dev = build_dpcn_model(device)
    load_weights_into_model(model, ckpt_path, dev)
    model.eval()
    return model, dev

@torch.no_grad()
def infer_seg_maps(model, x, *, fov=None, use_fov_in_model=False, threshold=0.5):
    """Return mask_u8, probs, logits, edge_probs, skel_probs (edge/skel may be None)."""
    use_amp = (x.device.type == "cuda")
    ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else torch.cpu.amp.autocast(enabled=False)
    with ctx:
        out = model(x, fov=fov) if use_fov_in_model else model(x)

    edge_probs = skel_probs = None
    if isinstance(out, dict):
        logits      = out.get("logits", None)
        edge_logits = out.get("edge_logits", None)
        skel_logits = out.get("skel_logits", None)
    elif isinstance(out, (tuple, list)):
        logits      = out[0]
        edge_logits = out[1] if len(out) > 1 else None
        skel_logits = out[2] if len(out) > 2 else None
    else:
        logits = out
        edge_logits = skel_logits = None

    if logits is None:
        raise RuntimeError("Model forward did not return main 'logits'.")

    probs = torch.sigmoid(logits)
    if fov is not None:
        probs = probs * (fov > 0.5).to(probs.dtype)
    pred01 = (probs >= threshold).to(torch.uint8)

    mask_u8 = pred01[0, 0].detach().cpu().numpy()
    if edge_logits is not None:
        edge_probs = torch.sigmoid(edge_logits)[0, 0].detach().cpu().numpy()
    if skel_logits is not None:
        skel_probs = torch.sigmoid(skel_logits)[0, 0].detach().cpu().numpy()
    return mask_u8, probs, logits, edge_probs, skel_probs
