import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

import numpy as np

def _resample_polyline_xy(xy, ds):
    """Uniform arc-length resample of an Nx2 polyline (x,y). Returns Mx2 points."""
    if len(xy) < 2: return xy.copy()
    seg = np.diff(xy, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = L[-1]
    if total < 1e-6: return xy[:1].copy()
    s = np.arange(0.0, total, ds)
    out = []
    j = 0
    for t in s:
        while j+1 < len(L) and L[j+1] < t:
            j += 1
        if j+1 >= len(L):
            out.append(xy[-1]); break
        # interpolate within segment j
        alpha = (t - L[j]) / max(L[j+1] - L[j], 1e-8)
        p = xy[j] + alpha * (xy[j+1] - xy[j])
        out.append(p)
    out = np.vstack(out)
    # always end at the last point
    if np.linalg.norm(out[-1] - xy[-1]) > 1e-6:
        out = np.vstack([out, xy[-1]])
    return out

def collect_orthogonal_chords(
    mask, graph, dist=None, *,
    stride_by_arc=3.0,         # tick spacing in *pixels of arc length*
    margin_from_nodes=8.0,     # skip within this many pixels from either end of an edge
    k_tangent=3.0,             # tangent window in pixels along the resampled curve
    step=0.25,                 # marching step (subpixel)
    max_radius=20.0,           # absolute cap for half-chord
    clip_k_edt=1.35,           # also cap half-chord <= k * EDT(center)
    min_len=0.5                # keep very thin chords
):
    """
    Returns chords as [((yL,xL),(yR,xR),(yc,xc)), ...]
    - Dense ticks (arc-length)
    - Skips near junctions and endpoints
    - Uses EDT to avoid overlong chords near branches
    """
    M = (mask > 0).astype(bool)
    H, W = M.shape
    have_edt = dist is not None

    def _full_path(e):
        return [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + \
               e["pixels"] + \
               [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]

    def _ray_endpoint(py, px, dy, dx, half_cap):
        """march from (py,px) along (dy,dx) ≤ half_cap (px), stop at mask boundary"""
        y, x = float(py), float(px)
        lasty, lastx = y, x
        r = 0.0
        while r < half_cap:
            y += dy * step; x += dx * step; r += step
            iy, ix = int(round(y)), int(round(x))
            if iy < 0 or iy >= H or ix < 0 or ix >= W: break
            if not M[iy, ix]:
                # step back one step for a crude boundary
                y -= dy * step; x -= dx * step
                break
            lasty, lastx = y, x
        return lasty, lastx

    chords = []
    for e in graph["edges"]:
        path_yx = _full_path(e)
        if len(path_yx) < 2: 
            continue

        # degree at endpoints (avoid branch zones)
        udeg = graph["nodes"][e["u"]]["deg"]
        vdeg = graph["nodes"][e["v"]]["deg"]

        # resample edge by arc length
        xy = np.array([(p[1], p[0]) for p in path_yx], dtype=np.float32)  # (x,y)
        xy_u = _resample_polyline_xy(xy, stride_by_arc)

        # cumulative length on resampled curve
        seg = np.diff(xy_u, axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        L = np.concatenate([[0.0], np.cumsum(seglen)])
        total = L[-1]

        # per-sample tangent via centered finite difference over ~k_tangent window
        for i in range(len(xy_u)):
            # skip near endpoints or if either endpoint is a junction (deg!=2)
            s = L[i]
            if s < margin_from_nodes or (total - s) < margin_from_nodes:
                continue
            if udeg != 2 or vdeg != 2:
                # whole edge touches a junction; still allow interior but keep larger margins
                if s < max(margin_from_nodes, 1.5*k_tangent) or (total - s) < max(margin_from_nodes, 1.5*k_tangent):
                    continue

            # indices for tangent window
            # find s±k_tangent in L
            s0 = max(0.0, s - k_tangent)
            s1 = min(total, s + k_tangent)
            # locate bracketing indices
            j0 = np.searchsorted(L, s0, side='left')
            j1 = np.searchsorted(L, s1, side='right') - 1
            if j1 <= j0:
                continue
            # approximate tangent as endpoint difference over window
            t = xy_u[j1] - xy_u[j0]
            nrm = np.hypot(t[0], t[1])
            if nrm < 1e-6:
                continue
            # normal = rotate tangent by 90°
            ny, nx = -t[1]/nrm, t[0]/nrm

            yc, xc = xy_u[i][1], xy_u[i][0]
            iy, ix = int(round(yc)), int(round(xc))

            # half-chord cap: min(max_radius, k * EDT)
            half_cap = max_radius
            if have_edt and 0 <= iy < H and 0 <= ix < W:
                half_cap = min(half_cap, clip_k_edt * float(dist[iy, ix]))

            yL, xL = _ray_endpoint(yc, xc, -ny, -nx, half_cap)
            yR, xR = _ray_endpoint(yc, xc,  ny,  nx, half_cap)
            length = np.hypot(yR - yL, xR - xL)
            if length >= min_len:
                chords.append(((yL, xL), (yR, xR), (yc, xc)))

    return chords


def plot_vessel_widths_overlay(
    rgb, mask, graph, *, chords=None,
    boundary_color="tab:blue", center_color="white", chord_color="red",
    # >>> 1px everywhere except 1.5px for red chords
    chord_lw=1.5, center_lw=1.0, boundary_lw=1.0,
    chord_alpha=0.95,
    max_chords=20000,           # allow plenty; we’ll still clip by zoom
    zoom=None,                  # zoom=(yc,xc,half_size) in pixels
    figsize=(7,7)
):
    """
    rgb: HxWx3 uint8
    mask: HxW {0,1}
    graph: from build_skeleton_graph
    chords: list[ ((yL,xL),(yR,xR),(yc,xc)), ... ]
    """
    H, W = mask.shape

    # view window
    y0, y1, x0, x1 = 0, H, 0, W
    if zoom is not None:
        yc, xc, half = zoom
        y0 = max(0, int(yc - half)); y1 = min(H, int(yc + half))
        x0 = max(0, int(xc - half)); x1 = min(W, int(xc + half))

    # prepare figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb[y0:y1, x0:x1]); ax.axis("off")

    # --- blue boundary (1 px) ---
    for c in find_contours(mask.astype(float), level=0.5):
        yy, xx = c[:, 0], c[:, 1]
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        if inside.any():
            ax.plot(xx[inside] - x0, yy[inside] - y0,
                    color=boundary_color, lw=boundary_lw, solid_capstyle="butt", antialiased=False)

    # --- white centerline (1 px) ---
    for e in graph["edges"]:
        path = [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + e["pixels"] + \
               [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]
        yy = np.array([p[0] for p in path]); xx = np.array([p[1] for p in path])
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        if inside.any():
            ax.plot(xx[inside] - x0, yy[inside] - y0,
                    color=center_color, lw=center_lw, solid_capstyle="butt", antialiased=False)

    # --- red width chords (1.5 px) ---
    if chords is None:
        raise ValueError("Pass chords from collect_orthogonal_chords_v2 (recommended).")

    # if too many, subsample uniformly
    if len(chords) > max_chords:
        idx = np.linspace(0, len(chords) - 1, max_chords).astype(int)
        chords_iter = (chords[i] for i in idx)
    else:
        chords_iter = chords

    # IMPORTANT: draw a chord if its *center* is inside the crop (endpoints may be outside)
    for (yL, xL), (yR, xR), (yc, xc) in chords_iter:
        if (y0 <= yc < y1) and (x0 <= xc < x1):
            ax.plot([xL - x0, xR - x0], [yL - y0, yR - y0],
                    color=chord_color, lw=chord_lw, alpha=chord_alpha,
                    solid_capstyle="butt", antialiased=False)

    ax.set_title("Centerline (white, 1px) • widths (red, 1.5px) • boundary (blue, 1px)")
    plt.show()
