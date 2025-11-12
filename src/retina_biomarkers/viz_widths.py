import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

import numpy as np

import numpy as np

def _bilinear_sample(arr, y, x):
    H, W = arr.shape
    if y < 0 or y > H-1 or x < 0 or x > W-1:
        return 0.0
    y0 = int(np.floor(y)); x0 = int(np.floor(x))
    y1 = min(y0+1, H-1);   x1 = min(x0+1, W-1)
    wy = y - y0; wx = x - x0
    v00 = arr[y0, x0]; v01 = arr[y0, x1]
    v10 = arr[y1, x0]; v11 = arr[y1, x1]
    return float((1-wy)*((1-wx)*v00 + wx*v01) + wy*((1-wx)*v10 + wx*v11))

def _resample_polyline_xy(xy, ds):
    if len(xy) < 2: return xy.copy()
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-6: return xy[:1].copy()
    ts = np.arange(0.0, total, ds)
    out = []
    j = 0
    for t in ts:
        while j+1 < len(s) and s[j+1] < t: j += 1
        if j+1 >= len(s):
            out.append(xy[-1]); break
        a = (t - s[j]) / max(s[j+1]-s[j], 1e-8)
        out.append(xy[j] + a*(xy[j+1]-xy[j]))
    if np.linalg.norm(out[-1]-xy[-1]) > 1e-6:
        out.append(xy[-1])
    return np.vstack(out)

def collect_orthogonal_chords_v3(
    mask, graph, dist, *,
    stride_by_arc=2.0,       # dense ticks
    margin_from_nodes=10.0,  # keep away from junctions/endpoints
    k_tangent=4.0,           # tangent window (px along arc)
    step=0.25,               # subpixel marching
    max_radius=20.0,         # hard cap
    clip_k_edt=1.25,         # also cap by k * local EDT
    min_len=0.5,             # accept very thin
    kappa_max=0.20,          # curvature gate (~1/radius); 0.2 => radius ~5px
    asym_max_ratio=1.6,      # reject chords with big Lleft/Lright asymmetry
    edt_eps=1e-3
):
    """
    Returns chords [((yL,xL),(yR,xR),(yc,xc)), ...] with:
      • arc-length sampling,
      • curvature gating,
      • EDT-monotone marching to first local maximum,
      • symmetry filtering to avoid diagonal overshoot near branches.
    """
    M = (mask > 0).astype(bool)
    H, W = M.shape

    def _full_path(e):
        return [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + \
               e["pixels"] + \
               [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]

    def _ray_to_local_max(yc, xc, ny, nx, half_cap):
        """march outward; stop at the first local EDT maximum or boundary."""
        y = float(yc); x = float(xc)
        lasty, lastx = y, x
        prev = _bilinear_sample(dist, y, x)
        grew = False
        r = 0.0
        while r < half_cap:
            y += ny*step; x += nx*step; r += step
            iy, ix = int(round(y)), int(round(x))
            if iy < 0 or iy >= H or ix < 0 or ix >= W or not M[iy, ix]:
                # step back one step for boundary touch
                y -= ny*step; x -= nx*step
                break
            cur = _bilinear_sample(dist, y, x)
            if cur + edt_eps < prev and grew:
                # passed the local max → backtrack one step
                y -= ny*step; x -= nx*step
                break
            grew |= (cur > prev + edt_eps)
            prev = cur
            lasty, lastx = y, x
        return lasty, lastx, r

    chords = []
    for e in graph["edges"]:
        path_yx = _full_path(e)
        if len(path_yx) < 2:
            continue
        udeg = graph["nodes"][e["u"]]["deg"]
        vdeg = graph["nodes"][e["v"]]["deg"]

        xy = np.array([(p[1], p[0]) for p in path_yx], dtype=np.float32)
        xy_u = _resample_polyline_xy(xy, stride_by_arc)
        seg = np.diff(xy_u, axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        S = np.concatenate([[0.0], np.cumsum(seglen)])
        total = S[-1]
        if total < 1e-6: 
            continue

        # precompute tangent angle along the resampled curve
        dxy = np.gradient(xy_u, axis=0)   # central diffs
        theta = np.arctan2(dxy[:,1], dxy[:,0])

        for i in range(len(xy_u)):
            s = S[i]
            # keep away from ends / junction touch
            if s < margin_from_nodes or (total - s) < margin_from_nodes:
                continue
            if udeg != 2 or vdeg != 2:
                if s < max(margin_from_nodes, 1.5*k_tangent) or (total - s) < max(margin_from_nodes, 1.5*k_tangent):
                    continue

            # curvature gate (|Δθ|/Δs) around i
            i0 = max(0, i-2); i1 = min(len(xy_u)-1, i+2)
            dth = np.unwrap(theta[i0:i1+1])
            Lwin = S[i1] - S[i0]
            if Lwin > 1e-6:
                kappa = abs(dth[-1] - dth[0]) / Lwin
                if kappa > kappa_max:
                    continue

            # normal
            t = dxy[i]; nrm = np.hypot(t[0], t[1])
            if nrm < 1e-6: 
                continue
            ny, nx = -t[1]/nrm, t[0]/nrm  # (y,x) normal

            yc, xc = xy_u[i][1], xy_u[i][0]
            iy, ix = int(round(yc)), int(round(xc))
            if iy < 0 or iy >= H or ix < 0 or ix >= W:
                continue

            half_cap = min(max_radius, clip_k_edt * float(dist[iy, ix]))

            yL, xL, rL = _ray_to_local_max(yc, xc, -ny, -nx, half_cap)
            yR, xR, rR = _ray_to_local_max(yc, xc,  ny,  nx, half_cap)

            length = np.hypot(yR - yL, xR - xL)
            if length < min_len:
                continue

            # symmetry check
            small = max(rL, 1e-6); big = max(rR, 1e-6)
            ratio = max(big, small) / max(min(big, small), 1e-6)
            if ratio > asym_max_ratio:
                continue

            chords.append(((yL, xL), (yR, xR), (yc, xc)))

    return chords



def plot_vessel_widths_overlay(
    rgb, mask, graph, *, chords=None,
    boundary_color="blue", center_color="white", chord_color="red",
    # >>> 1px everywhere except 1.5px for red chords
    chord_lw=1.0, center_lw=1.0, boundary_lw=1.0,
    chord_alpha=0.95,
    max_chords=30000,           # allow plenty; we’ll still clip by zoom
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

    ax.set_title("Centerline (white, 1px) • widths (red, 1px) • boundary (blue, 1px)")
    plt.show()
