import io
import numpy as np
import cv2
from PIL import Image
from src.data.preprocessing import _iso_resize_and_pad

def preprocess_image_retina_from_pil(
    pil_im: Image.Image,
    target_size: int = 512,
    use_gamma: bool = True,
    gamma: float = 0.9,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
) -> np.ndarray:
    g_u8 = np.array(pil_im.convert("RGB"), dtype=np.uint8)[..., 1]  # (H,W) uint8
    g_u8 = _iso_resize_and_pad(g_u8, target=target_size, pad_value=0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    g_eq_u8 = clahe.apply(g_u8)
    g = g_eq_u8.astype(np.float32) / 255.0
    if use_gamma and 0.5 <= gamma <= 1.2:
        g = np.power(g, gamma, dtype=np.float32)
    return np.expand_dims(g, axis=0).astype(np.float32)  # (1,H,W)

def preprocess_mask_from_bytes(mask_bytes: bytes, target_size: int = 512) -> np.ndarray:
    buf = np.frombuffer(mask_bytes, dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError("Could not decode mask bytes")
    if m.dtype != np.uint8:
        m = cv2.convertScaleAbs(m)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGRA2GRAY) if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = _iso_resize_and_pad(m, target=target_size, pad_value=0)
    m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    m = (m > 0).astype(np.float32)
    return np.expand_dims(m, axis=0).astype(np.float32)  # (1,H,W)

def load_fov_1hw_from_bytes(fov_bytes: bytes, target_size: int) -> np.ndarray:
    # Identical to preprocess_mask_from_bytes, just a clearer name for FOVs
    return preprocess_mask_from_bytes(fov_bytes, target_size)

def fov_bin_from_bytes(fov_bytes: bytes, out_wh: tuple[int, int]) -> np.ndarray:
    im = Image.open(io.BytesIO(fov_bytes)).convert("L").resize(out_wh, Image.NEAREST)
    arr = np.array(im, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)  # [H,W]
