import cv2
import numpy as np
from skimage import exposure

def _iso_resize_and_pad(img: np.ndarray, target: int = 512, pad_value: float = 0.0):
    h, w = img.shape[:2]
    scale = float(target) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    top = (target - nh) // 2
    bottom = target - nh - top
    left = (target - nw) // 2
    right = target - nw - left

    if img.ndim == 3:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=[pad_value]*3)
    else:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return padded

def _estimate_fov_mask(rgb_float01: np.ndarray):
    # crude, fast FOV: threshold on value channel then morph close
    hsv = cv2.cvtColor((rgb_float01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    thr = np.clip(cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1], 0, 255)
    thr = cv2.medianBlur(thr, 7)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((13,13), np.uint8))
    return (thr > 0).astype(np.float32)

def preprocess_image_retina(path: str, target_size: int = 512,
                            use_gamma: bool = True, gamma: float = 0.9,
                            clahe_clip: float = 2.0, clahe_tiles: int = 8,
                            apply_fov: bool = True) -> np.ndarray:
    # load BGR -> RGB float32 in [0,1]
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # isotropic resize + pad BEFORE intensity ops
    rgb = _iso_resize_and_pad(rgb, target=target_size, pad_value=0.0)

    # green channel dominance
    g = rgb[..., 1]

    # CLAHE on float via uint8 view, then back to float
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    g_eq = clahe.apply((g * 255.0).astype(np.uint8)).astype(np.float32) / 255.0

    # optional mild gamma
    if use_gamma and 0.5 <= gamma <= 1.2:
        g_eq = exposure.adjust_gamma(g_eq, gamma=gamma)

    # optional FOV mask to zero background
    if apply_fov:
        fov = _estimate_fov_mask(rgb)
        g_eq *= fov

    # channel-first, float32 [0,1]
    return np.expand_dims(g_eq.astype(np.float32), axis=0)

def preprocess_image_rgb(path: str, target_size: int = 512) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = _iso_resize_and_pad(rgb, target=target_size, pad_value=0.0)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)

def preprocess_mask(path: str, target_size: int = 512) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not load mask at {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = _iso_resize_and_pad(m, target=target_size, pad_value=0)
    m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    m = (m > 0).astype(np.float32)  # hard binary
    return np.expand_dims(m, axis=0).astype(np.float32)
