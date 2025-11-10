
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from .geometry import to_bool_mask
from .metrics_global import area_density, length_density, caliber_stats, tortuosity_stats, fractal_dimension_boxcount

def ring_masks_from_disc(shape: Tuple[int, int], disc_center: Tuple[float, float], PD_px: float, step_PD: float = 0.5, max_PD: float = 3.0):
    """
    Build boolean masks for concentric peripapillary rings centered at disc_center.
    disc_center: (y,x) in pixels. PD_px: optic disc diameter in pixels.
    Rings: [0, 0.5PD), [0.5PD, 1.0PD), ..., up to max_PD.
    Returns list of (inner_PD, outer_PD, mask).
    """
    H, W = shape
    cy, cx = disc_center
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - cy, xx - cx)
    rings = []
    d = PD_px
    inner = 0.0
    while inner < max_PD:
        outer = min(inner + step_PD, max_PD)
        mask = (r >= inner * d) & (r < outer * d)
        rings.append((inner, outer, mask))
        inner = outer
    return rings


def metrics_by_rings(mask: np.ndarray, graph: Dict, widths_per_edge, disc_center: Tuple[float, float], PD_px: float) -> Dict:
    """
    Compute a minimal set of metrics per 0.5-PD ring: area density, length density, and fractal dimension.
    """
    H, W = mask.shape
    rings = ring_masks_from_disc((H, W), disc_center, PD_px, step_PD=0.5, max_PD=3.0)
    results = {}
    for (r0, r1, roi) in rings:
        key = f"{r0:.1f}-{r1:.1f}PD"
        ad = area_density(mask, roi=roi)
        # length density restricted to ROI: approximate by counting edge pixels inside ROI
        skel_px = 0
        for e in graph["edges"]:
            for (y,x) in e["pixels"]:
                if roi[y, x]:
                    skel_px += 1
        ld = skel_px / float(H*W)
        fd = fractal_dimension_boxcount(to_bool_mask(mask) & roi)
        results[key] = {"area_density": ad, "length_density_px": ld, "fractal_dimension": fd}
    return results
