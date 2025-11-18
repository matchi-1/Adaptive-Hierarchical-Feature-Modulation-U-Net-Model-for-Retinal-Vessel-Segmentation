from tkinter import Image
import numpy as np
import cv2
from typing import Optional, Tuple
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from scipy.ndimage import binary_fill_holes
from src.data.preprocessing import _iso_resize_and_pad
from PIL import Image

def extract_disc_mask_safe(
    pred_map: np.ndarray,
    id2label: dict,
    img_shape=None,
    cup_dilate_frac: float = 0.10,
) -> np.ndarray:
    """
    Robust disc mask from a label map:
      1) try 'disc';
      2) else dilate 'cup';
      3) else use any non-background;
    then keep largest component + clean.
    Returns uint8 mask (0/1).
    """
    H, W = pred_map.shape if img_shape is None else img_shape[:2]
    name = {int(i): str(v).lower() for i, v in id2label.items()}

    disc_id = next((i for i, n in name.items() if ("disc" in n and "cup" not in n)), None)
    cup_id  = next((i for i, n in name.items() if "cup" in n), None)

    disc = np.zeros_like(pred_map, dtype=bool)
    if disc_id is not None:
        disc = (pred_map == disc_id)

    if disc.sum() == 0 and cup_id is not None:
        cup = (pred_map == cup_id)
        if cup.any():
            rad = max(5, int(cup_dilate_frac * min(H, W)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad+1, 2*rad+1))
            disc = cv2.dilate(cup.astype(np.uint8), k) > 0

    if disc.sum() == 0:
        disc = (pred_map != 0)

    lab = label(disc)
    if lab.max() == 0:
        return np.zeros_like(disc, dtype=np.uint8)
    areas = [(lab == i).sum() for i in range(1, lab.max()+1)]
    idx = 1 + int(np.argmax(areas))
    disc = (lab == idx)

    disc = binary_opening(disc, disk(3))
    disc = binary_closing(disc, disk(5))
    disc = binary_fill_holes(disc)
    disc = remove_small_objects(disc, min_size=200)
    return disc.astype(np.uint8)


def preprocess_image_retina_from_pil(
    pil_im: Image.Image,
    target_size: int = 512,
    use_gamma: bool = True,
    gamma: float = 0.9,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
) -> np.ndarray:
    g_u8 = np.array(pil_im.convert("RGB"), dtype=np.uint8)[..., 1]  # (H,W) uint8
    g_u8 = _iso_resize_and_pad(g_u8, target=target_size, pad_value=0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    g_eq_u8 = clahe.apply(g_u8)
    g = g_eq_u8.astype(np.float32) / 255.0
    if use_gamma and 0.5 <= gamma <= 1.2:
        g = np.power(g, gamma, dtype=np.float32)
    return np.expand_dims(g, axis=0).astype(np.float32)  # (1,H,W)


def center_and_pd_with_bounds(
    disc_bin,
    img_shape,
    r_frac=(0.08, 0.16),
    *,
    allow_fallback: bool = False,
    fallback_center_yx=None,   # e.g., (H/2, W/2)
    fallback_PD_px: float = None  # e.g., dataset constant or 0.2*min(H,W)
):
    H, W = img_shape[:2]
    disc_bin = (np.asarray(disc_bin) > 0)

    lab = label(disc_bin)
    if lab.max() == 0:
        if not allow_fallback:
            raise ValueError("center_and_pd_with_bounds: empty disc mask (no components).")
        # fallback center
        if fallback_center_yx is None:
            fallback_center_yx = (H/2.0, W/2.0)
        # fallback PD
        if fallback_PD_px is None:
            # ~radius 10% of min dim → PD ≈ 0.20 * min(H,W)
            fallback_PD_px = 0.20 * min(H, W)
        return fallback_center_yx, float(fallback_PD_px), float(fallback_PD_px)

    props = regionprops(lab.astype(int))[0]
    cy, cx = props.centroid
    area = float(props.area)
    PD_raw = 2.0 * np.sqrt(area / np.pi)

    r_min = r_frac[0] * min(H, W)
    r_max = r_frac[1] * min(H, W)
    PD_clamped = float(np.clip(PD_raw, 2*r_min, 2*r_max))
    return (float(cy), float(cx)), PD_raw, PD_clamped