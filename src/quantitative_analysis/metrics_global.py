
import numpy as np
from typing import Dict, List, Tuple, Optional
from .geometry import to_bool_mask

def area_density(mask: np.ndarray, roi: Optional[np.ndarray] = None) -> float:
    mask_bin = to_bool_mask(mask)
    if roi is not None:
        roi = to_bool_mask(roi)
        if roi.shape != mask_bin.shape:
            raise ValueError("ROI must match mask shape.")
        area_use = roi.sum()
        if area_use == 0:
            return 0.0
        return (mask_bin & roi).sum() / float(area_use)
    return mask_bin.mean()


def _edge_length_from_pixels(pixels: List[Tuple[int, int]]) -> float:
    if len(pixels) <= 1:
        return float(len(pixels))
    pts = np.array(pixels, dtype=np.float32)
    diffs = np.diff(pts[:, ::-1], axis=0)  # (x,y) diffs by reversing columns
    seglen = np.linalg.norm(diffs, axis=1)
    return float(seglen.sum())


def length_density(graph: Dict, image_shape: Tuple[int, int]) -> float:
    """
    Sum of edge lengths divided by image area (px^-1).
    """
    H, W = image_shape
    total_len = 0.0
    for e in graph["edges"]:
        total_len += _edge_length_from_pixels([(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + e["pixels"] + [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])])
    return total_len / float(H * W)


def caliber_stats(widths_per_edge: List[np.ndarray], bins: Tuple[float, float, float] = (4.0, 8.0, np.inf)) -> Dict[str, float]:
    """
    Compute global caliber statistics from width samples along skeleton edges.
    Returns median, IQR, and length-weighted thin/medium/thick fractions.
    bins: (thin_upper, medium_upper, thick_upper=inf) in pixels.
    """
    # Flatten
    all_widths = np.concatenate([w for w in widths_per_edge if w.size > 0], axis=0) if widths_per_edge else np.zeros(0, dtype=np.float32)
    stats = {"median_width": float(np.median(all_widths)) if all_widths.size else 0.0,
             "iqr_width": float(np.subtract(*np.percentile(all_widths, [75, 25]))) if all_widths.size else 0.0,
             "frac_thin_len": 0.0, "frac_med_len": 0.0, "frac_thick_len": 0.0}

    # Length-weighted fractions: approximate each sample as unit arc length
    # (If you have segment lengths, you can pass them; here we treat adjacent pixel steps similarly.)
    if all_widths.size:
        thin_u, med_u, thick_u = bins
        thin_mask = (all_widths <= thin_u)
        med_mask  = (all_widths > thin_u) & (all_widths <= med_u)
        thick_mask= (all_widths > med_u) & (all_widths <= thick_u)
        total = float(all_widths.size)
        stats["frac_thin_len"]  = float(thin_mask.sum()) / total
        stats["frac_med_len"]   = float(med_mask.sum())  / total
        stats["frac_thick_len"] = float(thick_mask.sum())/ total

    return stats


def _tortuosity_edge(pixels: List[Tuple[int, int]]) -> float:
    """
    Compute curvature-squared integral / length for one polyline edge (discrete).
    """
    if len(pixels) < 3:
        return 0.0
    pts = np.array([(x, y) for (y, x) in pixels], dtype=np.float32)  # to (x,y)
    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)  # length across segments
    # Guard against zero-length segments
    keep = seglen > 1e-6
    if keep.sum() < 2:
        return 0.0
    diffs = diffs[keep]
    seglen = seglen[keep]
    theta = np.arctan2(diffs[:, 1], diffs[:, 0])
    theta_u = np.unwrap(theta)
    dtheta = np.diff(theta_u)
    ds = seglen[1:]  # align with dtheta between successive segments
    ds[ds < 1e-6] = 1e-6
    kappa = np.abs(dtheta / ds)
    # Integral kappa^2 ds over path (Riemann sum)
    integral = float((kappa**2 * ds).sum())
    total_len = float(seglen.sum())
    if total_len < 1e-6:
        return 0.0
    return integral / total_len


def tortuosity_stats(graph: Dict) -> Dict[str, float]:
    """
    Tortuosity = length-weighted mean of per-edge (∫κ^2 ds)/L.
    """
    vals = []
    lens = []
    for e in graph["edges"]:
        # include endpoints in length computation for better geometry
        p = [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + e["pixels"] + [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]
        t = _tortuosity_edge(p)
        L = len(p)
        vals.append(t)
        lens.append(L)
    if not vals:
        return {"tortuosity_mean": 0.0}
    vals = np.asarray(vals, dtype=np.float32)
    lens = np.asarray(lens, dtype=np.float32)
    wmean = float((vals * lens).sum() / np.maximum(lens.sum(), 1e-6))
    return {"tortuosity_mean": wmean}


def fractal_dimension_boxcount(img: np.ndarray, min_box: int = 2) -> float:
    """
    Box-counting fractal dimension of a binary image (skeleton recommended).
    Returns the slope of log(N(s)) vs log(1/s) where s are powers of 2.
    """
    img = to_bool_mask(img)
    H, W = img.shape
    # choose box sizes as powers of 2 up to min(H,W)
    max_pow = int(np.floor(np.log2(min(H, W))))
    sizes = [2**k for k in range(max_pow, 0, -1) if 2**k >= min_box]
    if not sizes:
        return 0.0

    counts = []
    scales = []
    for s in sizes:
        # pad image to multiple of s
        pad_h = (s - (H % s)) % s
        pad_w = (s - (W % s)) % s
        pad_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
        HH, WW = pad_img.shape
        # reshape into blocks and count non-empty blocks
        blocks = pad_img.reshape(HH//s, s, WW//s, s).any(axis=(1,3))
        N = int(blocks.sum())
        counts.append(max(N, 1))
        scales.append(1.0 / float(s))

    # linear fit on logs
    x = np.log(np.asarray(scales, dtype=np.float64))
    y = np.log(np.asarray(counts, dtype=np.float64))
    if x.size < 2:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    # FD is slope
    return float(slope)
