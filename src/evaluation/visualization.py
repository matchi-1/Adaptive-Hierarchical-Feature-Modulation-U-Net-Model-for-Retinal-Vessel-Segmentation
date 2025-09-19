def visualize_predictions(model, dataset, img_paths=None, num_samples=3, threshold=0.5):
    """
    Shows rows of 4 columns:
      [Original Image | Preprocessed | Ground Truth | Predicted Mask]
    Handles tensors/ndarrays in CHW/HWC and grayscale/RGB.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _to_display_img(arr):
        """Return an image suitable for plt.imshow: (H,W) or (H,W,3/4), values normalized if needed."""
        # -> numpy
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)

        # Handle shape conversions (CHW -> HWC, drop singleton channel)
        if arr.ndim == 3:
            # If channels likely first
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            # If last channel is singleton, drop it to make grayscale
            if arr.shape[-1] == 1:
                arr = arr[..., 0]

        # Normalize floats or large integer ranges to [0,1] for safe display
        if arr.dtype.kind == 'f':
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            if mx > mn:
                arr = (arr - mn) / (mx - mn + 1e-8)
        elif arr.dtype.kind in 'iu':
            if arr.max(initial=0) > 255 or arr.min(initial=0) < 0:
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn + 1e-8)

        return arr

    def _to_display_mask(mask):
        """Return (H,W) mask as numpy."""
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            # Prefer squeezing channel-first or channel-last singleton
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                # If it's 3 channels, take one or convert; here we take first channel
                mask = mask[..., 0] if mask.shape[-1] in (3, 4) else mask[0]
        return mask

    model.eval()
    num_samples = min(num_samples, len(dataset))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]  # ensure 2D indexing

    for i in range(num_samples):
        # dataset should return: (image (tensor), true_mask (tensor), raw_img (np or tensor))
        image, true_mask, raw_img = dataset[i]

        # Forward pass
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
          logits = model(image_tensor)  # [1, C, H, W] where C=1 or C=2

          if logits.ndim == 4 and logits.shape[1] == 2:
              # 2-channel head (bg, vessel)
              probs_vessel = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # vessel prob in [0,1]
              probs = probs_vessel
          elif logits.ndim == 4 and logits.shape[1] == 1:
              # 1-channel head (vessel logit)
              probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
          else:
              # Fallback if model returns [2,H,W] or [H,W]
              if logits.shape[0] == 2:
                  probs = torch.softmax(logits, dim=0)[1].cpu().numpy()
              else:
                  probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_pred = (probs >= threshold).astype(np.uint8)

        # Prepare visuals
        raw_disp = _to_display_img(raw_img)
        pre_disp = _to_display_img(image)
        gt_disp = _to_display_mask(true_mask)

        # Plot
        # 1) Original
        ax = axes[i, 0]
        ax.imshow(raw_disp, cmap='gray' if raw_disp.ndim == 2 else None)
        ax.set_title("Original Image")
        ax.axis('off')

        # 2) Preprocessed
        ax = axes[i, 1]
        ax.imshow(pre_disp, cmap='gray' if pre_disp.ndim == 2 else None)
        ax.set_title("Preprocessed")
        ax.axis('off')

        # 3) Ground Truth
        ax = axes[i, 2]
        ax.imshow(gt_disp, cmap='gray')
        ax.set_title("Ground Truth")
        ax.axis('off')

        # 4) Predicted Mask
        ax = axes[i, 3]
        ax.imshow(binary_pred, cmap='gray')
        ax.set_title("Predicted Mask")
        ax.axis('off')

    plt.tight_layout()
    plt.show()