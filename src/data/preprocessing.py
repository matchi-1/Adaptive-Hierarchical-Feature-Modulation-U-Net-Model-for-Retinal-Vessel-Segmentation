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

def preprocess_image_retina(
    path: str,
    target_size: int = 512,                 # good alt: gamma=0.75, clahe_clip=3.5, clahe_tiles=4
    use_gamma: bool = True, gamma: float = 0.9,
    clahe_clip: float = 2.0, clahe_tiles: int = 8,
):
    """
    Memory-safe fundus preprocessing that works well for vessels.

    Sequence (intentionally ordered to avoid huge allocations):
      1) Read as BGR uint8 (cheap).
      2) Extract GREEN channel as uint8 (hemoglobin contrast lives here).
      3) Isotropic resize + pad the SINGLE channel to target size (still uint8).
      4) Apply CLAHE on uint8 (OpenCV expects 8-bit; avoids float32 3x memory).
      5) Convert to float32 in [0,1].
      6) Optional gentle gamma to lift faint vessels.
      7) Return (1,H,W) float32.

    Notes:
      - We DO NOT convert the full RGB to float32 before resizing — that’s what caused
        the large allocation (e.g., 5043×5837×3 float32 ≈ 337 MB).
      - FOV masking is handled later in the Dataset; keep it out of preprocessing.
    """

    # 1) Read image as BGR uint8 (no big intermediate arrays)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    # 2) Use the GREEN channel (uint8). This is cheap and vessel-friendly.
    #    (We *don’t* convert to float32 or RGB yet.)
    g_u8 = bgr[..., 1]  # shape (H,W), dtype=uint8

    # 3) Isotropic resize + pad to square target size (still uint8 to save memory).
    #    Ensure _iso_resize_and_pad handles 2D arrays and picks INTER_AREA for downscaling.
    g_u8 = _iso_resize_and_pad(g_u8, target=target_size, pad_value=0)

    # ----------------------------------------------------------------------
    # How CLAHE works:
    #   - Split into (H/tiles)×(W/tiles) grid of tiles.
    #   - Clip histogram bins to a cap derived from clipLimit to limit noise amp.
    #   - Normalize and interpolate across tiles for smooth transitions.
    # ----------------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))

    # 4) CLAHE expects 8-bit; apply in uint8 space for speed & stability.
    g_eq_u8 = clahe.apply(g_u8)  # still (H,W) uint8

    # 5) Convert to float32 [0,1] only after CLAHE
    g = g_eq_u8.astype(np.float32) / 255.0

    # 6) Optional gentle gamma (keep range checks to avoid weird config)
    if use_gamma and 0.5 <= gamma <= 1.2:
        # skimage.exposure.adjust_gamma expects float in [0,1]
        g = exposure.adjust_gamma(g, gamma=gamma)

    # 7) Return with a channel dimension: (1,H,W) float32
    return np.expand_dims(g.astype(np.float32), axis=0)

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



## -- HSI Intensity preprocessing variant -- ##

def _iso_resize_and_pad2(img: np.ndarray, target: int = 512, pad_value: float = 0.0, *, is_mask: bool = False):
    """
    Isotropic resize to fit the longer side to `target`, then symmetric pad to `target x target`.
    - For masks: nearest-neighbor.
    - For images: INTER_AREA when downscaling, INTER_LINEAR when upscaling.
    """
    h, w = img.shape[:2]
    scale = float(target) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    top = (target - nh) // 2
    bottom = target - nh - top
    left = (target - nw) // 2
    right = target - nw - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=(pad_value if img.ndim == 2 else [pad_value]*img.shape[2]))
    return padded


def preprocess_image_intensity_hsi(
    path: str,
    target_size: int = 512,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
    use_gamma: bool = True,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Same pipeline as preprocess_image_retina, but replaces G with HSI Intensity:
      1) Read BGR (uint8).
      2) Convert to RGB float [0,1], compute I = (R+G+B)/3.
      3) Resize+pad I in 8-bit with AREA/LINEAR (no nearest).
      4) CLAHE on I (uint8).
      5) To float32 [0,1]; optional gamma.
      6) Return (1, H, W) float32.

    Notes:
      - Masks/FOV handling unchanged elsewhere.
      - Recompute normalization stats for I over the train split.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    # RGB in [0,1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # HSI Intensity: average of channels (classic HSI definition)
    I = rgb.mean(axis=2)  # shape (H, W), float32 in [0,1]

    # Resize+pad in 8-bit for CLAHE efficiency
    I_u8 = (I * 255.0).astype(np.uint8)
    I_u8 = _iso_resize_and_pad2(I_u8, target=target_size, pad_value=0, is_mask=False)

    # CLAHE on intensity
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    I_eq_u8 = clahe.apply(I_u8)

    # Normalize to [0,1]
    I_f = I_eq_u8.astype(np.float32) / 255.0

    # Optional gentle gamma (helps lift faint vessels)
    if use_gamma and 0.5 <= gamma <= 1.2:
        I_f = exposure.adjust_gamma(I_f, gamma=gamma)

    return np.expand_dims(I_f.astype(np.float32), axis=0)  # (1, H, W)


def preprocess_image_mdfi_weighted(
    path: str,
    target_size: int = 512,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
    use_gamma: bool = True,
    gamma: float = 0.9,
    weights_rgb: tuple[float, float, float] = (0.2793, 0.7041, 0.0166),
) -> np.ndarray:
    """
    Preprocess fundus image using MDFI-Net's weighted grayscale:
        I_w = wR*R + wG*G + wB*B (weights sum to 1 after normalization).

    Steps (memory-safe, mirrors preprocess_image_retina):
      1) Read BGR (uint8) -> convert to RGB [0,1]
      2) Compute weighted grayscale I_w in float
      3) Resize+pad in 8-bit with AREA/LINEAR (no nearest)
      4) CLAHE on uint8
      5) Convert to float32 [0,1]
      6) Optional gamma
      7) Return (1,H,W) float32

    Notes:
      - Masks/FOV unchanged elsewhere.
      - If you tune weights, they will be renormalized to sum=1.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    # 1) BGR -> RGB in [0,1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 2) MDFI weighted grayscale (normalize weights defensively)
    wR, wG, wB = weights_rgb
    ws = wR + wG + wB
    if ws <= 0:
        raise ValueError("weights_rgb must have positive sum")
    wR, wG, wB = wR / ws, wG / ws, wB / ws

    Iw = wR * rgb[..., 0] + wG * rgb[..., 1] + wB * rgb[..., 2]  # float in [0,1]

    # 3) Resize+pad via 8-bit path for CLAHE speed/stability
    Iw_u8 = (Iw * 255.0 + 0.5).astype(np.uint8)
    Iw_u8 = _iso_resize_and_pad2(Iw_u8, target=target_size, pad_value=0, is_mask=False)

    # 4) CLAHE (local contrast)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    Iw_eq_u8 = clahe.apply(Iw_u8)

    # 5) To float [0,1]
    Iw_f = Iw_eq_u8.astype(np.float32) / 255.0

    # 6) Optional gentle gamma
    if use_gamma and 0.5 <= gamma <= 1.2:
        Iw_f = exposure.adjust_gamma(Iw_f, gamma=gamma)

    # 7) Add channel dim
    return np.expand_dims(Iw_f.astype(np.float32), axis=0)  # (1,H,W)

