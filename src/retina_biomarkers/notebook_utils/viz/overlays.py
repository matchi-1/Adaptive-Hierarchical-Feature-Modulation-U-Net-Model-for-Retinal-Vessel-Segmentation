import numpy as np, matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

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
    chord_lw=1.5,
    center_lw=1.0,
    boundary_lw=1.0,
    ring_color="cyan",
    zoom=None,                 # e.g. (cy, cx, r)
    figsize=(7.5, 7.5),
    # any chord-creation knobs can be passed through:
    **chord_kwargs
):
    """
    One-shot figure: OD + PD rings + vessel boundary (blue) + centerline (white) + width chords (red).
    Returns {'graph': ..., 'chords': ...} so you can reuse them.
    """
    rgb_iso   = out["rgb_iso"]
    mask      = out["pred_mask"]
    disc_mask = out["disc_mask"]
    center_yx = tuple(out["od"]["center_yx"])
    PD_px     = out["od"]["PD_px"]

    graph, chords = compute_graph_and_chords(mask, **chord_kwargs)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb_iso); ax.axis("off")
    ax.set_title("OD + PD rings + centerline (white) + boundary (blue) + widths (red)")

    # (a) boundary (blue, 1 px)
    for c in find_contours(mask.astype(float), level=0.5):
        yy, xx = c[:, 0], c[:, 1]
        ax.plot(xx, yy, color="blue", lw=boundary_lw, solid_capstyle="butt", antialiased=False)

    # (b) centerline (white, 1 px) from graph
    for e in graph["edges"]:
        u = graph["nodes"][e["u"]]; v = graph["nodes"][e["v"]]
        path = [(u["y"], u["x"])] + e["pixels"] + [(v["y"], v["x"])]
        yy = np.array([p[0] for p in path]); xx = np.array([p[1] for p in path])
        ax.plot(xx, yy, color="white", lw=center_lw, solid_capstyle="butt", antialiased=False)

    # (c) width chords (red, 1.5 px)
    for (yL, xL), (yR, xR), (yc, xc) in chords:
        ax.plot([xL, xR], [yL, yR], color="red", lw=chord_lw, alpha=0.95,
                solid_capstyle="butt", antialiased=False)

    # (d) OD contour + PD rings
    try:
        ax.contour(disc_mask.astype(bool), levels=[0.5], colors='yellow', linewidths=1.2)
    except Exception:
        pass
    _draw_pd_rings(ax, center_yx, PD_px, color=ring_color)

    # optional zoom window
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
