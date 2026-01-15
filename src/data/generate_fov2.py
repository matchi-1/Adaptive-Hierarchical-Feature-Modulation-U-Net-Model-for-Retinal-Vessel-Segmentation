import os
import numpy as np
from PIL import Image
import cv2  

def compute_fov_mask_from_black_bg(img_np, *, min_sum=1, smooth=True):
    """
    img_np: HxWxC uint8 fundus image (RGB or BGR, doesn't matter).
    Returns:
        fov_mask: HxW uint8 (0 or 255), where 255 = inside FOV.
    """
    # Ensure 3D
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    # Sum over channels; pure-black background will have sum == 0
    sum_ch = img_np.sum(axis=2)  # HxW
    raw_mask = sum_ch > min_sum  # bool

    # Convert to uint8 0/255
    fov_mask = (raw_mask.astype(np.uint8) * 255)

    if smooth:
        # Simple morphological closing + opening to clean edges
        # kernel size can be adjusted depending on your images
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel)
        fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, kernel)

        # Keep only the largest connected component (the main retinal disc),
        # in case there are stray non-zero pixels elsewhere.
        num_labels, labels = cv2.connectedComponents(fov_mask)
        if num_labels > 1:
            # Skip label 0 (background)
            areas = [(labels == i).sum() for i in range(1, num_labels)]
            largest_label = 1 + int(np.argmax(areas))
            fov_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return fov_mask


def make_fov_version(input_path, out_mask_path=None, out_img_path=None):
    # Load image
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    # Compute FOV mask
    fov_mask = compute_fov_mask_from_black_bg(img_np, min_sum=1, smooth=True)

    # Apply mask to image (keep same size)
    fov_mask_3 = np.stack([fov_mask] * 3, axis=-1)  # HxWx3
    img_fov = (img_np * (fov_mask_3 > 0)).astype(np.uint8)

    # Default output paths
    base, ext = os.path.splitext(input_path)
    if out_mask_path is None:
        out_mask_path = base + "_fov_mask.png"
    if out_img_path is None:
        out_img_path = base + "_fov_applied.png"

    # Save outputs
    Image.fromarray(fov_mask).save(out_mask_path)
    Image.fromarray(img_fov).save(out_img_path)

    return fov_mask, img_fov, out_mask_path, out_img_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute FOV mask from black background fundus image.")
    parser.add_argument("image_path", type=str, help="Path to fundus image")
    parser.add_argument("--out_mask", type=str, default=None, help="Output path for FOV mask PNG")
    parser.add_argument("--out_img", type=str, default=None, help="Output path for FOV-applied image PNG")
    args = parser.parse_args()

    fov_mask, img_fov, mpath, ipath = make_fov_version(
        args.image_path, out_mask_path=args.out_mask, out_img_path=args.out_img
    )
    print(f"Saved FOV mask to: {mpath}")
    print(f"Saved FOV-applied image to: {ipath}")
