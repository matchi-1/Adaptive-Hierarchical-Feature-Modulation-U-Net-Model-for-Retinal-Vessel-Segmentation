import cv2
import numpy as np
from skimage import exposure

'''
_iso_resize_and_pad
Purpose:
    - Isotropic resize to fit the longer side to `target`, then zero-pad to a square canvas.
    - Avoids aspect distortion that would bend thin vessels.
Inputs:
    - img: HxW[xC] image (uint8/float32). 2D mask or 3-channel RGB/BGR.
    - target: output side length (pixels).
    - pad_value: constant value for padding (0 for images/masks).
Outputs:
    - Padded image of shape target x target [x C], same dtype as input.
Notes:
    - Uses bilinear for images (ndim==3) and nearest for masks (ndim==2).
    - Keeps content centered; padding is split on both sides.
'''
def _iso_resize_and_pad(img: np.ndarray, target: int = 512, pad_value: float = 0.0):
    h, w = img.shape[:2]                                  # extract height and width ; discard channels
    scale = float(target) / max(h, w)                     # compute a scaling factor so the longer side (either H or W) becomes exactly target
    nh, nw = int(round(h * scale)), int(round(w * scale)) # apply the scale to height and width

    # img.ndim == 3: 3 dimensions (H × W × C) color image (RGB/BGR) -- raw fundus images
    # img.ndim == 2: 2  dimensions grayscale -- mask
        # INTER_LINEAR (bilinear): blends neighboring pixels smoothly
        # INTER_NEAREST: picks the closest pixel without blending
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST 
    resized = cv2.resize(img, (nw, nh), interpolation=interp)          # isotropic resize based on new dimensions

    # compute how much padding to add on each side:
        # ex: if new size is 512×307, we need 205 columns of padding
        # → Left = 102, Right = 103 (split symmetrically)

    # ensures content stays centered
    top = (target - nh) // 2                              # vertical padding (top)
    bottom = target - nh - top                            # vertical padding (bottom)
    left = (target - nw) // 2                             # horizontal padding (left)
    right = target - nw - left                            # horizontal padding (right)

    if img.ndim == 3:
        # constant-color pad for 3-channel images
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=[pad_value]*3) # [pad_value]*3 expands to [0.0, 0.0, 0.0] to match channels
    else:
        # constant-value pad for single-channel images/masks
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return padded


'''
_estimate_fov_mask
Purpose:
    Fast, approximate field-of-view (FOV) mask for retinal fundus images.
Method:
    - Convert RGB [0,1] to HSV.
    - Threshold the value channel to separate circular FOV from black borders.
    - Median blur and morphological close to fill small holes/gaps.
Inputs:
    rgb_float01: HxWx3 float32 RGB in [0,1].
Outputs:
    HxW float32 mask in {0.0, 1.0}.
Notes:
    This is a heuristic; precise FOVs can be obtained via circle detection if needed.
'''

def _estimate_fov_mask(rgb_float01: np.ndarray,
                       dataset: str | None = None) -> np.ndarray:
    
    if dataset and dataset.upper() == "STARE":
        return _estimate_fov_mask_stare(rgb_float01)
    
    hsv = cv2.cvtColor((rgb_float01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)  # convert RGB to HSV on uint8
    v = hsv[..., 2]                                          # value channel (brightness) 0 = hue, 1 = saturation, 2 = value
    thr = np.clip(cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1], 0, 255)  # create rough binary mask by a threshold of 10 in brightness
    thr = cv2.medianBlur(thr, 7)                             # remove salt-and-pepper noise ; for each pixel, look at its 7×7 neighborhood, sort the 49 values, take the median
    
    # MORPH_CLOSE = dilation followed by erosion
    # Structuring Element (SE): a 13×13 square (np.ones((13,13)))
        # Dilation - a black pixel becomes white if any white pixel is under the SE when it’s centered there
        # Erosion - a white pixel stays white only if the entire SE fits inside white
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((13,13), np.uint8))  # close small gaps
    return (thr > 0).astype(np.float32)                      # binary {0,1} mask


import cv2
import numpy as np

def _odd(n: int) -> int:
    """Return the nearest odd integer >= 3 for kernel sizes."""
    n = int(max(3, round(n)))
    return n if n % 2 == 1 else n + 1

def _keep_center_component(mask_u8: np.ndarray) -> np.ndarray:
    """
    Keep the connected component that contains the image center.
    Fallback to the largest non-background component if the center is background.
    mask_u8: HxW uint8 {0,255}
    """
    h, w = mask_u8.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    cx, cy = w // 2, h // 2
    center_label = labels[cy, cx]
    if center_label != 0:
        keep = center_label
    else:
        # Largest component (exclude label 0 = background)
        keep = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return (labels == keep).astype(np.uint8) * 255

def _estimate_fov_mask_stare(rgb_float01: np.ndarray,
                             border_frac: float = 0.08,
                             v_gauss_sigma: float = 1.2,
                             v_offset: float = 12.0,
                             close_frac: float = 0.04,
                             open_frac: float = 0.015) -> np.ndarray:
    """
    STARE-specific FOV:
      1) Compute HSV V-channel.
      2) Estimate background brightness from a border ring (green-ish, non-black).
      3) Threshold V at (border_median + v_offset).
      4) Clean with median/closing/opening.
      5) Keep center-connected component only.
    Returns float32 {0,1}.
    """
    # --- prep ---
    img8 = (rgb_float01 * 255).astype(np.uint8)
    h, w = img8.shape[:2]

    # --- V channel with light smoothing to handle vignetting ---
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    v_blur = cv2.GaussianBlur(v, (0, 0), v_gauss_sigma)

    # --- border ring sampling to estimate background brightness (not pure black in STARE) ---
    b = max(2, int(round(border_frac * min(h, w))))
    border_samples = np.concatenate([
        v_blur[:b, :].reshape(-1),
        v_blur[h-b:, :].reshape(-1),
        v_blur[:, :b].reshape(-1),
        v_blur[:, w-b:].reshape(-1),
    ], axis=0)

    bg_med = float(np.median(border_samples))  # robust against bright labels
    T = max(10.0, bg_med + v_offset)          # raise threshold above green border

    # --- threshold V against adaptive border-informed cutoff ---
    m = (v_blur > T).astype(np.uint8) * 255

    # --- quick sanity fallback: if almost everything is white, tighten threshold ---
    white_ratio = float(m.mean() / 255.0)
    if white_ratio > 0.98:
        # use a stricter cutoff based on 90th percentile of border to avoid full white mask
        bg_p90 = float(np.percentile(border_samples, 90))
        T2 = max(T, bg_p90 + v_offset)
        m = (v_blur > T2).astype(np.uint8) * 255

    # --- denoise + fill small gaps/holes ---
    k_med = _odd(0.012 * min(h, w))     # ~1.2% of min side, odd
    k_close = _odd(close_frac * min(h, w))
    k_open  = _odd(open_frac  * min(h, w))

    m = cv2.medianBlur(m, k_med)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((k_open,  k_open),  np.uint8))

    # --- keep center component only ---
    m = _keep_center_component(m)

    # --- final squeeze to {0,1} float32 ---
    return (m > 0).astype(np.float32)

def _estimate_fov_mask(rgb_float01: np.ndarray,
                       dataset: str | None = None) -> np.ndarray:
    """
    Default FOV estimator.
    - DRIVE/CHASE: original simple V>10 + median + closing.
    - STARE: use border-informed thresholding to handle green/dark borders.
    Returns float32 {0,1}.
    """
    if dataset and dataset.upper() == "STARE":
        return _estimate_fov_mask_stare(rgb_float01)

    # original fast path (DRIVE/CHASE)
    hsv = cv2.cvtColor((rgb_float01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    thr = cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1]
    thr = cv2.medianBlur(thr, 7)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((13,13), np.uint8))
    return (thr > 0).astype(np.float32)






'''
preprocess_image_retina
Purpose:
    Canonical preprocessing for retinal vessel segmentation using a single green channel:
      1) Load RGB, convert to float in [0,1].
      2) Isotropic resize + pad to square canvas.
      3) Extract green channel (best vessel contrast).
      4) CLAHE to enhance local contrast (conservative params).
      5) Optional mild gamma correction.
      6) Optional FOV masking to zero out background outside the circular fundus.
      7) Return (1, H, W) float32 in [0,1].
Inputs:
    path: image file path.
    target_size: output side length (pixels).
    use_gamma: enable/disable gamma correction.
    gamma: gamma exponent (<=1 brightens dark regions).
    clahe_clip: CLAHE clip limit.
    clahe_tiles: CLAHE tile grid size (square).
    apply_fov: zero background outside FOV.
Outputs:
    Numpy array shaped (1, target_size, target_size), float32 in [0,1].
Contracts:
    - No aspect distortion (isotropic scale).
    - No dtype ping-pong except where CLAHE requires uint8.
    - Background zeroed if apply_fov=True.
'''

def preprocess_image_retina(path: str, target_size: int = 512,
                            use_gamma: bool = True, gamma: float = 0.9,
                            clahe_clip: float = 2.0, clahe_tiles: int = 8,
                            apply_fov: bool = True) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)               # load image as BGR uint8
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")  # explicit failure if path is bad
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # convert to RGB float32 [0,1]

    rgb = _iso_resize_and_pad(rgb, target=target_size, pad_value=0.0)      # isotropic resize + zero pad

    g = rgb[..., 1]                                         # extract green channel (HxW float32 [0,1])

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))  # CLAHE op
    g_eq = clahe.apply((g * 255.0).astype(np.uint8)).astype(np.float32) / 255.0            # CLAHE on uint8 view

    if use_gamma and 0.5 <= gamma <= 1.2:                   # guardrails on gamma range
        g_eq = exposure.adjust_gamma(g_eq, gamma=gamma)     # mild gamma to lift faint vessels

    if apply_fov:
        fov = _estimate_fov_mask(rgb)                       # compute approximate FOV mask
        g_eq *= fov                                         # zero out background outside FOV

    return np.expand_dims(g_eq.astype(np.float32), axis=0)  # (1,H,W) float32 in [0,1]

'''
preprocess_image_rgb
Purpose:
    Preprocess a color image without channel reduction:
      - Load RGB in [0,1], isotropically resize + pad, return CHW float32.
Use cases:
    Visualization, auxiliary networks expecting 3 channels.
Inputs:
    path: image file path.
    target_size: output side length (pixels).
Outputs:
    (3, target_size, target_size) float32 in [0,1].
'''

def preprocess_image_rgb(path: str, target_size: int = 512) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)               # load BGR
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")  # fail fast
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # to RGB [0,1]
    rgb = _iso_resize_and_pad(rgb, target=target_size, pad_value=0.0)      # iso resize + pad
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)                  # HWC -> CHW float32

'''
preprocess_mask
Purpose:
    Prepare a binary segmentation mask aligned with the preprocessed images:
      - Load mask (any format), convert to grayscale if needed.
      - Isotropic resize + pad with nearest-neighbor.
      - Otsu threshold to hard binary {0,1}.
      - Return (1,H,W) float32.
Inputs:
    path: mask file path.
    target_size: output side length (pixels).
Outputs:
    (1, target_size, target_size) float32 with values in {0.0, 1.0}.
Notes:
    - Nearest-neighbor is used for geometry to avoid label bleeding.
    - Otsu ensures deterministic hard labels after resizing.
'''

def preprocess_mask(path: str, target_size: int = 512) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)             # load mask as-is (uint8 or palette)
    if m is None:
        raise FileNotFoundError(f"Could not load mask at {path}")  # fail fast
    if m.ndim == 3:                                        # if mask is color/palette, convert to gray
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = _iso_resize_and_pad(m, target=target_size, pad_value=0)    # iso resize + pad (nearest)
    m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]    # hard threshold to {0,255}
    m = (m > 0).astype(np.float32)                         # cast to {0.0, 1.0}
    return np.expand_dims(m, axis=0).astype(np.float32)    # (1,H,W) float32
