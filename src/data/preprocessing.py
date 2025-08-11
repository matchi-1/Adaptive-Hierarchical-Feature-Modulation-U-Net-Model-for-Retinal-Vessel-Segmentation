import cv2
import numpy as np
from skimage import exposure

'''
_iso_resize_and_pad
Purpose:
    Isotropic resize to fit the longer side to `target`, then zero-pad to a square canvas.
Rationale:
    Avoids aspect distortion that would bend thin vessels.
Inputs:
    img: HxW[xC] image (uint8/float32). 2D mask or 3-channel RGB/BGR.
    target: output side length (pixels).
    pad_value: constant value for padding (0 for images/masks).
Outputs:
    Padded image of shape target x target [x C], same dtype as input.
Notes:
    - Uses bilinear for images (ndim==3) and nearest for masks (ndim==2).
    - Keeps content centered; padding is split on both sides.
'''
def _iso_resize_and_pad(img: np.ndarray, target: int = 512, pad_value: float = 0.0):
    h, w = img.shape[:2]                                  # current height and width
    scale = float(target) / max(h, w)                     # scale so the longer side becomes `target`
    nh, nw = int(round(h * scale)), int(round(w * scale)) # new integer dimensions after scaling
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST  # choose interpolation based on channels
    resized = cv2.resize(img, (nw, nh), interpolation=interp)          # isotropic resize

    top = (target - nh) // 2                              # symmetric vertical padding (top)
    bottom = target - nh - top                            # symmetric vertical padding (bottom)
    left = (target - nw) // 2                             # symmetric horizontal padding (left)
    right = target - nw - left                            # symmetric horizontal padding (right)

    if img.ndim == 3:
        # constant-color pad for 3-channel images
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=[pad_value]*3)
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

def _estimate_fov_mask(rgb_float01: np.ndarray):
    hsv = cv2.cvtColor((rgb_float01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)  # convert to HSV on uint8
    v = hsv[..., 2]                                          # value channel (brightness)
    thr = np.clip(cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1], 0, 255)  # simple low threshold
    thr = cv2.medianBlur(thr, 7)                             # remove salt-and-pepper noise
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((13,13), np.uint8))  # close small gaps
    return (thr > 0).astype(np.float32)                      # binary {0,1} mask


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
