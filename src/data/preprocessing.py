import cv2
import numpy as np
from skimage import exposure

def preprocess_image_clahe(image_path, resize=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    image = cv2.resize(image, resize)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    weights = np.array([0.2793, 0.7041, 0.0166])
    image_gray = np.dot(image_rgb[..., :3], weights).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=5.0)
    image_clahe = clahe.apply(image_gray)

    image_gamma = exposure.adjust_gamma(image_clahe, gamma=0.8)
    image_gamma = np.clip(image_gamma, 0, 255).astype(np.uint8)

    image = image_gamma / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def preprocess_image_rgb(image_path, resize=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    image = cv2.resize(image, resize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return image

def preprocess_mask(mask_path, resize=(512, 512)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at {mask_path}")
    mask = cv2.resize(mask, resize)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=0).astype(np.float32)
    return mask
