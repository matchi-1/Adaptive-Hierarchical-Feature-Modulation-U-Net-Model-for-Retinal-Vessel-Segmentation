import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_refuge_segformer(
    model_name: str = "pamixsun/segformer_for_optic_disc_cup_segmentation",
    device: Optional[str] = None,
):
    """
    Returns (processor, model, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    return processor, model, device

def infer_label_map(
    rgb: np.ndarray,
    processor: AutoImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Runs the model and returns HxW uint8 class map (0=bg, 'disc', 'cup' IDs per model config).
    """
    if device is None:
        device = next(model.parameters()).device
    H, W = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, C, h, w]
    ups = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = ups.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred

def run_od(
    image_path: str,
    *,
    processor: Optional[AutoImageProcessor] = None,
    model: Optional[SegformerForSemanticSegmentation] = None,
    device: Optional[str] = None,
    cup_dilate_frac: float = 0.08,
    r_frac = (0.08, 0.16),
    return_intermediates: bool = False,
):
    """
    Convenience: path -> (rgb, pred_map, disc_mask, (cy,cx), PD_px_clamped[, PD_px_raw]).
    """
    from .postproc import extract_disc_mask_safe, center_and_pd_with_bounds
    rgb = imread_rgb(image_path)
    if (processor is None) or (model is None):
        processor, model, device = load_refuge_segformer(device=device)
    pred = infer_label_map(rgb, processor, model, device=device)

    disc_mask = extract_disc_mask_safe(pred, model.config.id2label, img_shape=rgb.shape, cup_dilate_frac=cup_dilate_frac)
    if disc_mask.sum() == 0:
        raise RuntimeError("No optic disc detected after fallbacks.")

    center_yx, PD_raw, PD_px = center_and_pd_with_bounds(disc_mask, rgb.shape, r_frac=r_frac)
    if return_intermediates:
        return rgb, pred, disc_mask, center_yx, PD_px, PD_raw
    return rgb, pred, disc_mask, center_yx, PD_px
