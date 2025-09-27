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
derive_fov_mask_path_from_image
Purpose:
    - Construct the corresponding FOV mask file path from a given image path.
    - Follows the dataset’s naming/layout convention across DRIVE, CHASEDB1, and STARE.
Inputs:
    - image_path: str. Full path to an image file inside the `images/` folder.
Outputs:
    - str. Full path to the corresponding mask file inside the `mask/` folder.
Notes:
    - Rule: images/01_training.jpg -> mask/01_training_mask.png
    - Rule: images/01_test.jpg     -> mask/01_test_mask.png
    - Works uniformly for DRIVE, CHASEDB1, and STARE datasets after renaming
'''
def derive_fov_mask_path_from_image(image_path: str) -> str:
    p = Path(image_path)                    # convert to Path object
    # replace 'images' directory with 'mask'. / "mask" appends a child folder named "mask" to that directory, yielding .../training/mask
    mask_dir = p.parent.parent / "mask"     # p.parent is the directory containing the file. ".parent.parent" moves it from /training/images to /training
    stem = p.stem                            # p.stem is the file name without its extension.e.g., "01_training" or "02_test"
    mask_name = f"{stem}_mask.png"           # append "_mask", force extension .png
    return str(mask_dir / mask_name)         # joins the mask_dir from step 2 with the mask_name from step 4 to form the full mask path



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
                            mask_path: str | None = None): #auto_discover_mask: bool = True) -> np.ndarray
    

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
        cand: Path | None = None

        # *** STRICT MODE: FOV mask must be provided and must exist. ***
        if mask_path is None:
            raise ValueError(
                "FOV mask_path is required but was not provided. "
            )
        cand = Path(mask_path)

        if not cand.exists():
            raise FileNotFoundError(f"FOV mask not found at {cand}")

        # preprocess the existing mask to align geometry
        fov_mask = preprocess_mask(str(cand), target_size=target_size)[0]  # (1,H,W)->(H,W)

        if fov_mask is not None:
            g_eq *= fov_mask  # elementwise gating

    return np.expand_dims(g_eq.astype(np.float32), axis=0)  # (1,H,W) float32 in [0,1]



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

