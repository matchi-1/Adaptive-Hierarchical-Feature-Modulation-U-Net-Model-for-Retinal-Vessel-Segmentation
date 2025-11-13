import numpy as np, matplotlib.pyplot as plt

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
