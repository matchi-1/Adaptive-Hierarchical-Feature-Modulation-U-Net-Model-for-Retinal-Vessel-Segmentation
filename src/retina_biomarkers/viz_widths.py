import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# ---------- 1) Build orthogonal chords ----------
def collect_orthogonal_chords(
    mask, graph, *,
    k_tangent=3,      # how far ahead/behind to estimate tangent
    step=0.5,         # ray-march step (px)
    max_radius=20.0,  # half-chord cap
    stride=5,         # sample every N-th skeleton pixel to reduce clutter
    min_len=2.0       # drop super short chords
):
    """
    Returns a list of chords: [((yL,xL),(yR,xR),(yc,xc)), ...]
    where (yc,xc) is the skeleton point and (yL,xL)-(yR,xR) are
    the two boundary intersections along the normal direction.
    """
    M = (mask > 0).astype(bool)
    H, W = M.shape

    def _full_path(graph, e):
        return [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + \
               e["pixels"] + \
               [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]

    def _unit(v):
        n = np.hypot(v[0], v[1])
        return v / (n + 1e-8)

    def _ray_endpoint(y, x, dy, dx):
        """march from (y,x) along (dy,dx) until leaving mask or max_radius"""
        py, px = float(y), float(x)
        last_py, last_px = py, px
        r = 0.0
        while r < max_radius:
            py += dy * step; px += dx * step; r += step
            iy, ix = int(round(py)), int(round(px))
            if iy < 0 or iy >= H or ix < 0 or ix >= W:
                break
            if not M[iy, ix]:
                # step back one step for a crude boundary estimate
                py -= dy * step; px -= dx * step
                break
            last_py, last_px = py, px
        return last_py, last_px

    chords = []
    for e in graph["edges"]:
        path = _full_path(graph, e)
        if len(path) < (2*k_tangent + 1):
            continue
        for i in range(0, len(path), stride):
            y, x = path[i]
            # tangent using a centered finite difference
            i0 = max(0, i - k_tangent)
            i1 = min(len(path)-1, i + k_tangent)
            y0, x0 = path[i0]; y1, x1 = path[i1]
            ty, tx = float(y1 - y0), float(x1 - x0)
            if abs(ty) + abs(tx) < 1e-6:
                continue
            # normal (perpendicular): rotate tangent by 90°
            ny, nx = -tx, ty
            ny, nx = _unit(np.array([ny, nx], dtype=np.float32))

            yL, xL = _ray_endpoint(y, x, -ny, -nx)
            yR, xR = _ray_endpoint(y, x,  ny,  nx)
            # filter tiny chords
            length = np.hypot(yR - yL, xR - xL)
            if length >= min_len:
                chords.append(((yL, xL), (yR, xR), (y, x)))

    return chords


# ---------- 2) Draw: boundary (blue), centerline (white), width chords (red) ----------
def plot_vessel_widths_overlay(
    rgb, mask, graph, *, chords=None,
    boundary_color="tab:blue", center_color="white", chord_color="red",
    chord_alpha=0.9, chord_lw=1.3, center_lw=1.5, boundary_lw=1.5,
    max_chords=1200, zoom=None,  # zoom=(yc,xc,half_size) for tight crop
    figsize=(7,7)
):
    """
    rgb: HxWx3 uint8 (ISO frame recommended)
    mask: HxW {0,1}
    graph: from build_skeleton_graph
    chords: optional precomputed chords (else computed with defaults)
    zoom: set to (yc,xc,half) to crop a close-up like your example
    """
    H, W = mask.shape
    if chords is None:
        chords = collect_orthogonal_chords(mask, graph, stride=5, k_tangent=3, step=0.5, max_radius=20.0)

    # optional crop
    y0, y1, x0, x1 = 0, H, 0, W
    if zoom is not None:
        yc, xc, half = zoom
        y0 = max(0, int(yc - half)); y1 = min(H, int(yc + half))
        x0 = max(0, int(xc - half)); x1 = min(W, int(xc + half))

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.imshow(rgb[y0:y1, x0:x1]); ax.axis("off")

    # blue boundary
    for c in find_contours(mask.astype(float), level=0.5):
        yy, xx = c[:,0], c[:,1]
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        if inside.any():
            ax.plot(xx[inside]-x0, yy[inside]-y0, color=boundary_color, lw=boundary_lw)

    # white centerline (draw each edge as a polyline)
    for e in graph["edges"]:
        path = [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + e["pixels"] + \
               [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]
        yy = np.array([p[0] for p in path]); xx = np.array([p[1] for p in path])
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        if inside.any():
            ax.plot(xx[inside]-x0, yy[inside]-y0, color=center_color, lw=center_lw)

    # red chords (subset if too many)
    if len(chords) > max_chords:
        idx = np.linspace(0, len(chords)-1, max_chords).astype(int)
        chords_to_draw = [chords[i] for i in idx]
    else:
        chords_to_draw = chords

    for (yL,xL),(yR,xR),(yc,xc) in chords_to_draw:
        if (y0 <= yL < y1 and x0 <= xL < x1) or (y0 <= yR < y1 and x0 <= xR < x1):
            ax.plot([xL-x0, xR-x0], [yL-y0, yR-y0], color=chord_color, lw=chord_lw, alpha=chord_alpha)

    ax.set_title("Centerline (white), widths (red), boundary (blue)")
    plt.show()
