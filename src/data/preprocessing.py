import cv2
import numpy as np
from skimage import exposure
from pathlib import Path

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
_estimate_fov_mask   -- used in generate_fov.py to generate FOV masks for CHASEDB1 dataset (DRIVE and STARE already have FOV masks available)
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


'''
preprocess_image_retina
Purpose:
    Canonical preprocessing for retinal vessel segmentation using a single green channel:
      1) Load RGB, convert to float in [0,1].
      2) Isotropic resize + pad to square canvas.
      3) Extract green channel (best vessel contrast).
      4) CLAHE to enhance local contrast (conservative params).
      5) Mild gamma correction.
      6) FOV masking to zero out background outside the circular fundus.
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

def preprocess_image_retina(path: str,
                            target_size: int = 512,                    # another good observable value: gamma=0.75, clahe_clip=3.5, clahe_tiles=4
                            use_gamma: bool = True, gamma: float = 0.9,
                            clahe_clip: float = 2.0, clahe_tiles: int = 8,
                            apply_fov: bool = True,
                            mask_path: str | None = None,
                            auto_discover_mask: bool = True) -> np.ndarray:
    

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)               # load image as BGR uint8
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")  # explicit failure if path is bad
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # convert to RGB float32 [0,1]

    rgb = _iso_resize_and_pad(rgb, target=target_size, pad_value=0.0)      # isotropic resize + zero pad

    g = rgb[..., 1]                                         # extract green channel (HxW float32 [0,1])


    '''
    How CLAHE works:
        - Split the image into a grid of tiles of size (H/clahe_tiles)×(W/clahe_tiles)
        - For each tile, compute its 256-bin histogram
        - Clip each bin to a cap T derived from clipLimit to prevent rare bins from exploding contrast (noise amplification).
        - Normalize the histogram so it sums to 1
        - Interpolate between tiles to smooth the transitions
    '''

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))  # CLAHE op

    # why green channel: hemoglobin absorbs green → vessels have stronger contrast in G than R/B; dropping to one channel reduces noise and parameters.
    g_eq = clahe.apply((g * 255.0).astype(np.uint8)).astype(np.float32) / 255.0            # CLAHE on uint8 view

    if use_gamma and 0.5 <= gamma <= 1.2:                   # guardrails on gamma range
        g_eq = exposure.adjust_gamma(g_eq, gamma=gamma)     # mild gamma to lift faint vessels

    # --- FOV gating: prefer existing mask; else estimator if allowed ---
    if apply_fov:
        fov_mask = None

        # discover mask path if not explicitly provided
        cand = None
        if mask_path is not None:
            cand = Path(mask_path)
        elif auto_discover_mask:
            cand = _infer_mask_path(path)

        if cand is not None and cand.exists():
            # preprocess the existing mask to align geometry
            fov_mask = preprocess_mask(str(cand), target_size=target_size)[0]  # (1,H,W)->(H,W)
        else:
            # fall back when no file exists
            fov_mask = _estimate_fov_mask(rgb)

        if fov_mask is not None:
            g_eq *= fov_mask  # elementwise gating

    return np.expand_dims(g_eq.astype(np.float32), axis=0)  # (1,H,W) float32 in [0,1]

'''
preprocess_mask
Purpose:
    Load an existing FOV (or label) mask and align it to the model canvas.
    - Accepts 0/255, 0/1, or arbitrary grayscale; thresholds to {0,1}.
    - Geometry uses isotropic resize + pad (nearest) to avoid label bleed.
    - Returns (1, H, W) float32 in {0.0, 1.0}.
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
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not load mask at {path}")

    # collapse to single channel if needed
    if m.ndim == 3:
        # handle palettized/color masks robustly
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # ensure uint8 for stable thresholding
    if m.dtype != np.uint8:
        # normalize any numeric dtype into 0..255 range defensively
        m = m.astype(np.float32)
        # if already in {0,1}, scale up
        if m.max() <= 1.0:
            m = (m * 255.0)
        # clip and cast
        m = np.clip(m, 0, 255).astype(np.uint8)

    # geometry: iso resize + pad (nearest inside helper due to 2D input)
    m = _iso_resize_and_pad(m, target=target_size, pad_value=0)

    # binarize robustly:
    # - if histogram is strongly bimodal, Otsu works
    # - otherwise, any nonzero is treated as foreground
    # try Otsu first
    _, otsu = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # fallback union in case of weird grayscale: nonzero wins
    nz = (m > 0).astype(np.uint8) * 255
    m_bin = cv2.bitwise_or(otsu, nz)

    m_bin = (m_bin > 0).astype(np.float32)
    return np.expand_dims(m_bin, axis=0).astype(np.float32)


def _infer_mask_path(image_path: str | Path) -> Path:
    """
    Infer mask path from an image path by swapping 'images' -> 'mask' and forcing .png.
    Works for DRIVE/CHASEDB1/STARE layouts like .../<split>/images/<name>.<ext>
    """
    p = Path(image_path)
    parts = list(p.parts)
    try:
        i = parts.index("images")
        parts[i] = "mask"
    except ValueError:
        # fallback: put mask alongside
        return p.with_suffix(".png")
    return Path(*parts).with_suffix(".png")




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
        raise FileNotFoundError(f"Could not load image at {path}")  
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # to RGB [0,1]
    rgb = _iso_resize_and_pad(rgb, target=target_size, pad_value=0.0)      # iso resize + pad
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)                  # HWC -> CHW float32
