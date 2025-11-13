# regional.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from .geometry import to_bool_mask, _edge_full_path
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

def _roi_area(roi):  # boolean mask
    return float(roi.sum())

def _edge_length_in_roi(graph, e, roi):
    path = _edge_full_path(graph, e)
    if len(path) < 2: return 0.0
    import numpy as np
    pts = np.asarray([(p[1],p[0]) for p in path], dtype=np.float32)  # (x,y)
    inside = np.array([bool(roi[y,x]) for (y,x) in path], dtype=bool)
    L = 0.0
    for i in range(len(path)-1):
        if inside[i] or inside[i+1]:
            d = np.linalg.norm(pts[i+1]-pts[i])
            L += float(d)
    return L

# regional.py
def _edge_length_in_roi_fraction(p0, p1, roi, samples=7):
    # fraction of the segment [p0->p1] that lies inside roi
    (y0, x0), (y1, x1) = p0, p1
    ts = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    ys = (1 - ts) * y0 + ts * y1
    xs = (1 - ts) * x0 + ts * x1
    inside = 0
    H, W = roi.shape
    for y, x in zip(ys, xs):
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < H and 0 <= ix < W and roi[iy, ix]:
            inside += 1
    return inside / samples

def _total_edge_length_in_roi(graph, roi, samples=7):
    total = 0.0
    for e in graph["edges"]:
        path = _edge_full_path(graph, e)
        if len(path) < 2: 
            continue
        pts = np.asarray([(p[1], p[0]) for p in path], dtype=np.float32)  # (x,y)
        for i in range(len(path) - 1):
            seg_len = float(np.linalg.norm(pts[i+1] - pts[i]))
            frac = _edge_length_in_roi_fraction(path[i], path[i+1], roi, samples=samples)
            total += seg_len * frac
    return total


def _edge_length_in_roi_sampled(graph, e, roi, samples=7):
    path = _edge_full_path(graph, e)
    if len(path) < 2:
        return 0.0
    pts = np.asarray([(p[1], p[0]) for p in path], dtype=np.float32)  # (x,y)
    L_in = 0.0
    for i in range(len(path) - 1):
        seg_len = float(np.linalg.norm(pts[i+1] - pts[i]))
        frac = _edge_length_in_roi_fraction(path[i], path[i+1], roi, samples=samples)
        L_in += seg_len * frac
    return L_in


def _tortuosity_edge(pixels):
    if len(pixels) < 3: return 0.0
    import numpy as np
    pts = np.array([(x,y) for (y,x) in pixels], dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    keep = seglen > 1e-6
    if keep.sum() < 2: return 0.0
    diffs = diffs[keep]; seglen = seglen[keep]
    th = np.unwrap(np.arctan2(diffs[:,1], diffs[:,0]))
    dth = np.diff(th); ds = seglen[1:]; ds[ds<1e-6]=1e-6
    kappa = np.abs(dth/ds)
    integral = float((kappa**2 * ds).sum())
    return integral / float(seglen.sum())


def _tortuosity_mean_in_roi(graph, roi):
    vals, lens = [], []
    for e in graph["edges"]:
        path = _edge_full_path(graph, e)
        t = _tortuosity_edge(path)                      # tortuosity on full path
        L_in = _edge_length_in_roi_sampled(graph, e, roi, samples=9)  # ← sampled length *inside* ROI
        if L_in > 0:
            vals.append(t); lens.append(L_in)
    if not vals:
        return 0.0
    vals = np.asarray(vals, np.float32); lens = np.asarray(lens, np.float32)
    return float((vals * lens).sum() / max(lens.sum(), 1e-6))


def _width_samples_in_roi(widths_per_edge, graph, roi, stride=1, *, edt_interior=True):
    """
    Collect width samples that fall in roi.
    If edt_interior=True, widths_per_edge[i] corresponds to e["pixels"] (interior only).
    If edt_interior=False, widths_per_edge[i] corresponds to the full path ([u] + pixels + [v]) with same stride.
    """
    sel = []
    for w, e in zip(widths_per_edge, graph["edges"]):
        path = _edge_full_path(graph, e)  # [u] + pixels + [v]
        if len(path) == 0 or w.size == 0:
            continue

        if edt_interior:
            # widths index 0 ↔ path index 1, ..., widths index n-1 ↔ path index n
            for k in range(0, len(e["pixels"]), max(1, stride)):
                path_idx = k + 1
                y, x = path[path_idx]
                if roi[y, x]:
                    sel.append(w[k])
        else:
            # widths aligned to full path indices directly
            for idx in range(0, min(len(path), len(w)), max(1, stride)):
                y, x = path[idx]
                if roi[y, x]:
                    sel.append(w[idx])

    return np.asarray(sel, dtype=np.float32) if sel else np.zeros(0, dtype=np.float32)

def metrics_by_rings(mask, graph, widths_per_edge, disc_center, PD_px,
                     use_orth=True, *, skel=None, edt_interior=True):
    H, W = mask.shape
    if skel is None:
        from .geometry import skeletonize_mask
        skel = skeletonize_mask(mask)
    rings = ring_masks_from_disc((H, W), disc_center, PD_px, step_PD=0.5, max_PD=3.0)
    out = {}
    for (r0, r1, roi) in rings:
        key = f"{r0:.1f}-{r1:.1f}PD"
        area = _roi_area(roi)
        if area == 0:
            out[key] = {"area_density": 0.0, "length_density": 0.0, "fractal_dimension": 0.0,
                        "median_width": 0.0, "iqr_width": 0.0, "tortuosity_mean": 0.0}
            continue
        ad = area_density(mask, roi=roi)
        total_len_in = _total_edge_length_in_roi(graph, roi, samples=9)   # ← sampled version
        ld = total_len_in / area

        fd = fractal_dimension_boxcount(skel & roi)            # <--- skeleton!
        ws = _width_samples_in_roi(widths_per_edge, graph, roi, stride=1, edt_interior=edt_interior)
        if ws.size:
            med = float(np.median(ws)); iqr = float(np.subtract(*np.percentile(ws, [75, 25])))
        else:
            med = 0.0; iqr = 0.0
        tt = _tortuosity_mean_in_roi(graph, roi)
        out[key] = {"area_density": ad, "length_density": ld, "fractal_dimension": fd,
                    "median_width": med, "iqr_width": iqr, "tortuosity_mean": tt}
    return out

def quadrant_masks(shape, center):
    H, W = shape; cy, cx = center
    yy, xx = np.mgrid[0:H, 0:W]
    return {
        "upper":  (yy <  cy),
        "lower":  (yy >= cy),
        "left":   (xx <  cx),
        "right":  (xx >= cx),
    }


def widths_from_chords(chords, roi=None):
    out = []
    for (yL, xL), (yR, xR), (yc, xc) in chords:
        if roi is not None:
            iy, ix = int(round(yc)), int(round(xc))
            if not (0 <= iy < roi.shape[0] and 0 <= ix < roi.shape[1] and roi[iy, ix]):
                continue
        out.append(float(np.hypot(yR - yL, xR - xL)))
    return np.asarray(out, dtype=np.float32) if out else np.zeros(0, dtype=np.float32)
