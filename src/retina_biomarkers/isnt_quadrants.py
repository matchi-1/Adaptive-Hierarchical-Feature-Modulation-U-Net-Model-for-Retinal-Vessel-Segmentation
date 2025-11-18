# src/retina_biomarkers/isnt_quadrants.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patheffects


from src.data.preprocessing import _iso_resize_and_pad
from src.retina_biomarkers.od_seg import (
    load_refuge_segformer, infer_label_map, extract_disc_mask_safe,
    center_and_pd_with_bounds
)

try:
    from src.retina_biomarkers.metrics import compute_biomarkers_from_mask_array 
except Exception:
    compute_biomarkers_from_mask_array = None


# =========================
# I/O helpers
# =========================
def _read_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def _read_fov_mask(path, ref_shape):
    """
    Returns a boolean mask (H, W) where True = inside FOV.
    If path is None or missing, returns an all-True mask with ref_shape.
    """
    if (path is None) or (not os.path.exists(path)):
        return np.ones(ref_shape[:2], dtype=bool)
    m = Image.open(path)
    if m.mode != "L":
        m = m.convert("L")
    m = np.array(m, dtype=np.uint8)
    if m.shape != ref_shape[:2]:
        m = np.array(Image.fromarray(m).resize((ref_shape[1], ref_shape[0]), resample=Image.NEAREST))
    return m > 0

def find_fovea_xy_from_cyan(fovea_img_path, ref_shape=None, tol=40):
    """
    Detect the (x, y) of your cyan marker (#00FFF0-ish) and scale to ref_shape if provided.
    Returns (fx, fy) in pixels.
    """
    f = _read_rgb(fovea_img_path)
    Hf, Wf = f.shape[:2]
    R, G, B = f[...,0], f[...,1], f[...,2]
    mask = (R <= tol) & (G >= 255 - tol) & (B >= 240 - tol)
    if not mask.any():
        # broader fallback
        mask = (R < 80) & (G > 170) & (B > 170)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError(f"[fovea] cyan dot not found in {fovea_img_path}")
    fy, fx = float(ys.mean()), float(xs.mean())
    if ref_shape and (ref_shape[0] != Hf or ref_shape[1] != Wf):
        fy *= ref_shape[0] / Hf
        fx *= ref_shape[1] / Wf
    return fx, fy  # (x, y)


# =========================
# Optic disc localization 
# =========================
IMAGE_SIZE      = 512
R_FRAC          = (0.08, 0.16)
CUP_DILATE_FRAC = 0.12

# Load once
_processor, _od_model, _od_device = load_refuge_segformer()

def od_from_image_path(color_path):
    """
    Returns:
      rgb_iso  : (H,W,3) ISO-resized RGB
      disc_mask: (H,W) bool
      center_yx: (y, x) float
      PD_px    : float (roughly sized if exact not needed)
    """
    fundus_pil = Image.open(color_path).convert("RGB")
    rgb_iso = _iso_resize_and_pad(np.array(fundus_pil), target=IMAGE_SIZE, pad_value=0).astype(np.uint8)
    pred_map = infer_label_map(rgb_iso, _processor, _od_model, device=_od_device)
    disc_mask = extract_disc_mask_safe(pred_map, _od_model.config.id2label, img_shape=rgb_iso.shape,
                                       cup_dilate_frac=CUP_DILATE_FRAC)

    ys, xs = np.nonzero(disc_mask.astype(bool))
    if ys.size == 0:
        H, W = rgb_iso.shape[:2]
        center_yx, _, PD_px = center_and_pd_with_bounds(
            disc_mask, rgb_iso.shape, r_frac=R_FRAC,
            allow_fallback=True, fallback_center_yx=(H/2.0, W/2.0),
            fallback_PD_px=0.20 * min(H, W)
        )
    else:
        cy = float(ys.mean()); cx = float(xs.mean())
        center_yx = (cy, cx)
        H, W = rgb_iso.shape[:2]
        PD_px = 0.20 * min(H, W)  # rough PD ok for geometry/visuals

    return rgb_iso, disc_mask, center_yx, float(PD_px)


# =========================
# ISNT quadrant masks (disc→fovea anchored)
# =========================
def _wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi  # (-pi, pi]

def isnt_quadrants_masks(shape_hw, disc_yx, *, fovea_yx, sector_deg=90.0, fov=None):
    """
    Build four exclusive wedges (T/I/N/S) centered on angles:
      T: 0°, I: +90°, N: ±180°, S: -90° in the disc→fovea frame.
    Returns dict of boolean masks keyed by 'T','I','N','S'.
    """
    H, W = shape_hw
    cy, cx = disc_yx
    fy, fx = fovea_yx

    # v = Temporal direction (toward fovea), p = +90° (Inferior dir in image coords)
    v = np.array([fx - cx, fy - cy], float)
    v /= (np.linalg.norm(v) + 1e-9)
    p = np.array([-v[1], v[0]])

    yy, xx = np.mgrid[0:H, 0:W]
    relx, rely = (xx - cx).astype(float), (yy - cy).astype(float)

    # coordinates in (v,p)
    a = relx*v[0] + rely*v[1]
    b = relx*p[0] + rely*p[1]
    phi = np.arctan2(b, a)  # angle relative to v

    base = np.ones((H, W), bool) if fov is None else fov.astype(bool)

    half = np.deg2rad(sector_deg / 2.0)
    eps  = 1e-6
    centers = {"T": 0.0, "I": np.pi/2, "N": np.pi, "S": -np.pi/2}

    dT = np.abs(_wrap_pi(phi - centers["T"]))
    dI = np.abs(_wrap_pi(phi - centers["I"]))
    dN = np.abs(_wrap_pi(phi - centers["N"]))
    dS = np.abs(_wrap_pi(phi - centers["S"]))

    cand_T = (dT <= half + eps) & base
    cand_I = (dI <= half + eps) & base
    cand_N = (dN <= half + eps) & base
    cand_S = (dS <= half + eps) & base

    masks, taken = {}, np.zeros((H, W), dtype=bool)
    for key, cand in (("T", cand_T), ("I", cand_I), ("N", cand_N), ("S", cand_S)):
        m = cand & (~taken)
        masks[key] = m
        taken |= m

    return masks


# =========================
# Per-quadrant biomarker computation
# =========================
def compute_biomarkers_per_quadrant(
    pred_mask_u8, disc_center_yx, PD_px, quadrant_masks,
    *, compute_fn=None, **compute_kwargs
):
    """
    Compute your biomarker dict per quadrant by masking the vessel map.
    - pred_mask_u8: (H,W) uint8/bool vessel mask
    - quadrant_masks: dict {'T','I','N','S'} -> bool mask
    - compute_fn: callable(img_mask_u8, disc_center=..., PD_px=..., **kwargs)
                  defaults to `compute_biomarkers_from_mask_array` if available
    Returns: dict like {'T': biom_dict, 'I': biom_dict, ...}
    """
    fn = compute_fn or compute_biomarkers_from_mask_array
    if fn is None:
        raise RuntimeError("No biomarker computation function available. Pass compute_fn=...")

    out = {}
    for k, m in quadrant_masks.items():
        submask = (pred_mask_u8.astype(bool) & m).astype(np.uint8)
        out[k] = fn(
            submask,
            disc_center=disc_center_yx,
            PD_px=PD_px,
            **compute_kwargs
        )
    return out


# =========================
# Visualization
# =========================
def _label(ax, x, y, txt, fs=12):
    pe = [patheffects.withStroke(linewidth=3, foreground='black', alpha=0.6)]
    t = ax.text(x, y, txt, color="white", fontsize=fs, ha="center", va="center")
    t.set_path_effects(pe)

def draw_isnt_quadrants(ax, img_rgb, masks_dict, disc_yx, fovea_yx):
    ax.imshow(img_rgb); ax.axis("off")

    colors = {"I": (0.10, 0.70, 1.00, 0.22),   # cyan
              "S": (1.00, 0.25, 0.65, 0.22),   # magenta
              "N": (1.00, 0.90, 0.10, 0.22),   # yellow
              "T": (0.10, 0.85, 0.35, 0.22)}   # green

    overlay = np.zeros((*img_rgb.shape[:2], 4), float)
    for k, m in masks_dict.items():
        overlay[m, :3] = colors[k][:3]
        overlay[m, 3]  = colors[k][3]
    ax.imshow(overlay)

    cy, cx = disc_yx
    fy, fx = fovea_yx
    v = np.array([fx - cx, fy - cy], float)
    v /= (np.linalg.norm(v) + 1e-9)
    p = np.array([-v[1], v[0]])
    H, W = img_rgb.shape[:2]; L = max(H, W)

    # draw 45° boundaries (X)
    d1 = ( v + p) / np.sqrt(2)
    d2 = (-v + p) / np.sqrt(2)
    d3 = (-v - p) / np.sqrt(2)
    d4 = ( v - p) / np.sqrt(2)
    for d in (d1, d2, d3, d4):
        ax.plot([cx - L*d[0], cx + L*d[0]], [cy - L*d[1], cy + L*d[1]], color="white", lw=1)

    # markers
    ax.scatter([cx], [cy], s=50, c="yellow", marker="x", linewidths=1.5)  # disc
    ax.scatter([fx], [fy], s=50, c="cyan",   marker="x", linewidths=1.5)  # fovea

    # label each sector via centroid (robust under FOV clipping)
    def _centroid(m, fallback_xy):
        ys, xs = np.nonzero(m)
        if ys.size == 0: return fallback_xy
        return (float(xs.mean()), float(ys.mean()))

    cx_, cy_ = cx, cy
    for key, name in (("T","Temporal"), ("N","Nasal"), ("S","Superior"), ("I","Inferior")):
        x_, y_ = _centroid(masks_dict[key], (cx_, cy_))
        _label(ax, x_, y_, name)


# =========================
# High-level helper for a single image
# =========================
def prepare_isnt_for_image(
    color_path, fovea_img_path, fov_path=None, sector_deg=90.0
):
    """
    Returns dict with:
      'rgb_iso', 'disc_center_yx', 'PD_px', 'fovea_yx',
      'fov_mask', 'isnt_masks'
    """
    rgb_iso, disc_mask, center_yx, PD_px = od_from_image_path(color_path)
    fov_mask = _read_fov_mask(fov_path, rgb_iso.shape)
    fx, fy = find_fovea_xy_from_cyan(fovea_img_path, ref_shape=rgb_iso.shape[:2])
    fovea_yx = (fy, fx)
    isnt_masks = isnt_quadrants_masks(
        rgb_iso.shape[:2], center_yx, fovea_yx=fovea_yx,
        sector_deg=sector_deg, fov=fov_mask
    )
    return {
        "rgb_iso": rgb_iso,
        "disc_center_yx": center_yx,
        "PD_px": PD_px,
        "fovea_yx": fovea_yx,
        "fov_mask": fov_mask,
        "isnt_masks": isnt_masks,
    }
