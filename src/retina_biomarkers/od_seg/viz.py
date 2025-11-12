import numpy as np
import matplotlib.pyplot as plt

def show_disc_overlay(rgb, disc_mask, center_yx, PD_px, show_rings=True):
    cy, cx = center_yx
    theta = np.linspace(0, 2*np.pi, 512)
    plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.contour(disc_mask, levels=[0.5], colors='r', linewidths=1.5)
    plt.scatter([cx], [cy], s=40, marker='x')
    if show_rings:
        for k in np.arange(0.5, 3.0 + 1e-6, 0.5):
            r = k * PD_px
            yy = cy + r*np.sin(theta); xx = cx + r*np.cos(theta)
            plt.plot(xx, yy, linewidth=1)
        r = 2.0 * PD_px
        plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), '--', linewidth=1)
    plt.title(f"OD center=({cy:.1f},{cx:.1f}) • PD={PD_px:.1f}px")
    plt.axis("off"); plt.show()

def show_triptych(rgb, pred_map, disc_mask, center_yx, PD_px):
    cy, cx = center_yx
    theta = np.linspace(0, 2*np.pi, 512)
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20')
    vis_map = cmap(pred_map % 20)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(rgb); plt.axis('off'); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(vis_map); plt.axis('off'); plt.title("Segmentation (OD/OC)")
    plt.subplot(1,3,3); plt.imshow(rgb)
    plt.contour(disc_mask, levels=[0.5], colors='r', linewidths=1.5)
    plt.scatter([cx], [cy], s=30, marker='x')
    for k in np.arange(0.5, 3.0 + 1e-6, 0.5):
        r = k * PD_px
        yy = cy + r*np.sin(theta); xx = cx + r*np.cos(theta)
        plt.plot(xx, yy, linewidth=1)
    r = 2.0 * PD_px
    plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), '--', linewidth=1)
    plt.axis('off'); plt.title("OD + PD rings")
    plt.tight_layout(); plt.show()
