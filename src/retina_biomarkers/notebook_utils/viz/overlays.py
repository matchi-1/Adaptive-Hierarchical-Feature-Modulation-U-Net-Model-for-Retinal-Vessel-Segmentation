import numpy as np, matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

from skimage.segmentation import find_boundaries
from skimage.draw import line as draw_line


from src.retina_biomarkers import (
    skeletonize_mask, build_skeleton_graph, distance_transform,
    collect_orthogonal_chords
)

def _draw_pd_rings(ax, center_yx, PD_px, *, max_PD=3.0, step_PD=0.5, color='cyan'):
    if center_yx is None or PD_px is None: return
    cy, cx = center_yx
    theta = np.linspace(0, 2*np.pi, 512)
    ax.scatter([cx], [cy], s=36, marker='x', color='yellow')
    for k in np.arange(step_PD, max_PD + 1e-9, step_PD):
        r = k * PD_px
        xx = cx + r*np.cos(theta); yy = cy + r*np.sin(theta)
        ax.plot(xx, yy, color=color, linewidth=1.2)
    # dashed 2-PD
    r = 2.0 * PD_px
    xx = cx + r*np.cos(theta); yy = cy + r*np.sin(theta)
    ax.plot(xx, yy, color=color, linestyle="--", linewidth=1.4)

def compute_graph_and_chords(
    mask,
    *,
    stride_by_arc=1.0,
    margin_from_nodes=16.0,
    step=0.25,
    max_radius=20.0,
    clip_k_edt=1.25,
    min_len=0.75,
    tan_win_px=11,
    use_pca=True,
    pca_radius_px=7,
    kappa_max=0.18,
    asym_max_ratio=1.6,
):
    """
    Build skeleton graph and sample orthogonal width chords with robust defaults.
    Returns (graph, chords).
    """
    skel  = skeletonize_mask(mask)
    graph = build_skeleton_graph(skel)
    dist  = distance_transform(mask)

    chords = collect_orthogonal_chords(
        mask, graph, dist,
        stride_by_arc=stride_by_arc,
        margin_from_nodes=margin_from_nodes,
        step=step,
        max_radius=max_radius,
        clip_k_edt=clip_k_edt,
        min_len=min_len,
        tan_win_px=tan_win_px,
        use_pca=use_pca,
        pca_radius_px=pca_radius_px,
        kappa_max=kappa_max,
        asym_max_ratio=asym_max_ratio,
    )
    return graph, chords

def visualize_widths_centerline_boundary(
    out: dict,
    *,
    pixel_exact=True,           # <<< NEW: draw as 1-pixel raster overlays
    overlay_alpha=1.0,
    chord_lw=1.5,               # used only when pixel_exact=False
    center_lw=1.0,              # used only when pixel_exact=False
    boundary_lw=1.0,            # used only when pixel_exact=False
    ring_color="cyan",
    ring_style="vector",        # "vector" or "pixel"
    zoom=None,                  # e.g. (cy, cx, r)
    figsize=(7.5, 7.5),
    **chord_kwargs              # forwarded to compute_graph_and_chords
):
    rgb_iso   = out["rgb_iso"]
    mask      = out["pred_mask"]
    disc_mask = out["disc_mask"]
    center_yx = tuple(out["od"]["center_yx"])
    PD_px     = out["od"]["PD_px"]

    graph, chords = compute_graph_and_chords(mask, **chord_kwargs)

    H, W = mask.shape
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Always keep the base image crisp to the pixel grid
    ax.imshow(rgb_iso, interpolation="nearest")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("OD + PD rings + centerline (white) + boundary (blue) + widths (red)")

    if pixel_exact:
        # Build a single RGBA overlay in image pixel space
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        # (1) boundary as 1-pixel blue
        bmask = find_boundaries(mask.astype(bool), mode="outer")
        overlay[bmask] = [0, 0, 255, 255]

        # (2) centerline as 1-pixel white
        for e in graph["edges"]:
            u = graph["nodes"][e["u"]]; v = graph["nodes"][e["v"]]
            path = [(u["y"], u["x"])] + e["pixels"] + [(v["y"], v["x"])]
            for (yy, xx) in path:
                iy, ix = int(yy), int(xx)
                if 0 <= iy < H and 0 <= ix < W:
                    overlay[iy, ix] = [255, 255, 255, 255]  # white
        # (3) chords as 1-pixel red (draw last so they appear on top)
        for (yL, xL), (yR, xR), _ in chords:
            rr, cc = draw_line(int(round(yL)), int(round(xL)), int(round(yR)), int(round(xR)))
            good = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            overlay[rr[good], cc[good]] = [255, 0, 0, 255]

        # (4) disc contour (yellow) as 1-pixel boundary
        try:
            dmask = find_boundaries(disc_mask.astype(bool), mode="outer")
            # draw disc after boundary/centerline? up to you; here we draw over them
            overlay[dmask] = [255, 255, 0, 255]
        except Exception:
            pass

        ax.imshow(overlay, interpolation="nearest", alpha=overlay_alpha)

        # PD rings: vector by default; or raster circles if you want 1-px too
        if ring_style == "vector":
            _draw_pd_rings(ax, center_yx, PD_px, color=ring_color)
        else:
            # simple pixel rings: radii rounded to nearest pixel
            cy, cx = center_yx
            theta = np.linspace(0, 2*np.pi, 1024)
            for k in np.arange(0.5, 3.0 + 1e-9, 0.5):
                r = int(round(k * PD_px))
                xx = (cx + r*np.cos(theta)).round().astype(int)
                yy = (cy + r*np.sin(theta)).round().astype(int)
                inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                ring_overlay = np.zeros((H, W, 4), dtype=np.uint8)
                ring_overlay[yy[inb], xx[inb]] = [0, 255, 255, 255]  # cyan
                ax.imshow(ring_overlay, interpolation="nearest", alpha=1.0)
    else:
        # Original vector drawing (will not be exact pixel width)
        from skimage.measure import find_contours
        for c in find_contours(mask.astype(float), level=0.5):
            yy, xx = c[:, 0], c[:, 1]
            ax.plot(xx, yy, color="blue", lw=boundary_lw, solid_capstyle="butt", antialiased=False)

        for e in graph["edges"]:
            u = graph["nodes"][e["u"]]; v = graph["nodes"][e["v"]]
            path = [(u["y"], u["x"])] + e["pixels"] + [(v["y"], v["x"])]
            yy = np.array([p[0] for p in path]); xx = np.array([p[1] for p in path])
            ax.plot(xx, yy, color="white", lw=center_lw, solid_capstyle="butt", antialiased=False)

        for (yL, xL), (yR, xR), _ in chords:
            ax.plot([xL, xR], [yL, yR], color="red", lw=chord_lw, alpha=0.95,
                    solid_capstyle="butt", antialiased=False)

        try:
            ax.contour(disc_mask.astype(bool), levels=[0.5], colors='yellow', linewidths=1.2)
        except Exception:
            pass
        _draw_pd_rings(ax, center_yx, PD_px, color=ring_color)

    if zoom is not None:
        cy, cx, r = zoom
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy + r, cy - r)

    plt.tight_layout()
    plt.show()
    return {"graph": graph, "chords": chords}

def _draw_pd_rings(ax, center_yx, PD_px, *, max_PD=3.0, step_PD=0.5, color='cyan'):
    if center_yx is None or PD_px is None: return
    cy, cx = center_yx
    theta = np.linspace(0, 2*np.pi, 512)
    ax.scatter([cx], [cy], s=36, marker='x', color='yellow')
    for k in np.arange(step_PD, max_PD + 1e-9, step_PD):
        r = k * PD_px
        xx = cx + r*np.cos(theta); yy = cy + r*np.sin(theta)
        ax.plot(xx, yy, color=color, linewidth=1.2)
    # dashed 2-PD
    r = 2.0 * PD_px
    xx = cx + r*np.cos(theta); yy = cy + r*np.sin(theta)
    ax.plot(xx, yy, color=color, linestyle="--", linewidth=1.4)

def visualize_grid(out):
    raw_rgb     = out["raw_rgb"]
    rgb_iso     = out["rgb_iso"]
    disc_mask   = out["disc_mask"]
    center_yx   = tuple(out["od"]["center_yx"])
    PD_px       = out["od"]["PD_px"]
    pre_gray_u8 = out["pre_gray_u8"]
    prob_map    = out["prob_map"]
    mask_thr05  = out["pred_mask_thr05"]

    fig, axs = plt.subplots(2, 3, figsize=(14, 9)); axs = axs.ravel()
    axs[0].imshow(raw_rgb); axs[0].set_title("Raw"); axs[0].axis('off')
    axs[1].imshow(rgb_iso); axs[1].set_title("ISO-resized"); axs[1].axis('off')
    axs[2].imshow(rgb_iso); axs[2].set_title("OD + PD rings"); axs[2].axis('off')
    try: axs[2].contour(disc_mask.astype(bool), levels=[0.5], colors='r', linewidths=1.2)
    except: pass
    _draw_pd_rings(axs[2], center_yx, PD_px)

    axs[3].imshow(pre_gray_u8, cmap='gray', vmin=0, vmax=255); axs[3].set_title("CLAHE+gamma"); axs[3].axis('off')
    im5 = axs[4].imshow(prob_map, vmin=0, vmax=1); axs[4].set_title("Prob map"); axs[4].axis('off')
    plt.colorbar(im5, ax=axs[4], fraction=0.046, pad=0.04)
    axs[5].imshow(mask_thr05, cmap='gray', vmin=0, vmax=1); axs[5].set_title("Seg @0.5"); axs[5].axis('off')
    plt.tight_layout(); plt.show()

def visualize_overlay(out, *, alpha=0.35):
    rgb_iso    = out["rgb_iso"]
    disc_mask  = out["disc_mask"]
    center_yx  = tuple(out["od"]["center_yx"])
    PD_px      = out["od"]["PD_px"]
    mask_thr05 = out["pred_mask_thr05"]

    overlay = np.zeros_like(rgb_iso, dtype=np.uint8)
    overlay[..., 1] = (mask_thr05.astype(np.uint8) * 255)
    plt.figure(figsize=(6.5, 6.5))
    plt.imshow(rgb_iso); plt.imshow(overlay, alpha=alpha); plt.axis('off'); plt.title("Vessels + OD + PD rings")
    try: plt.contour(disc_mask.astype(bool), levels=[0.5], colors='r', linewidths=1.2)
    except: pass
    _draw_pd_rings(plt.gca(), center_yx, PD_px, color='white')
    plt.show()
