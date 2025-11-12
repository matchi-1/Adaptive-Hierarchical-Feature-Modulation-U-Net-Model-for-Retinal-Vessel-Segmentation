import numpy as np
import cv2
from typing import Optional, Tuple
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from scipy.ndimage import binary_fill_holes

def extract_disc_mask_safe(
    pred_map: np.ndarray,
    id2label: dict,
    img_shape=None,
    cup_dilate_frac: float = 0.06,
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

def center_and_pd_with_bounds(
    disc_bin: np.ndarray,
    img_shape,
    r_frac: Tuple[float, float] = (0.08, 0.16),
):
    """
    Compute center (cy,cx) and PD from disc area; clamp PD to a plausible radius fraction.
    Returns (center_yx, PD_raw, PD_clamped).
    """
    H, W = img_shape[:2]
    lab = label(disc_bin)
    if lab.max() == 0:
        return None, None, None
    props = regionprops(lab.astype(int))[0]
    cy, cx = props.centroid
    area = float(props.area)
    PD_raw = 2.0 * np.sqrt(area / np.pi)

    r_min = r_frac[0] * min(H, W)
    r_max = r_frac[1] * min(H, W)
    PD_clamped = float(np.clip(PD_raw, 2*r_min, 2*r_max))
    return (float(cy), float(cx)), PD_raw, PD_clamped
