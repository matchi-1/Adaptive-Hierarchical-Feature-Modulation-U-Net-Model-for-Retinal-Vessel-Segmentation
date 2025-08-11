import cv2
import numpy as np
from skimage import exposure

'''
 implements the three-step preprocessing you described in the paper: 
     1. RGB → weighted grayscale
     2. CLAHE
     3. gamma correction

  returns a single-channel image shaped (1, H, W) with float32 values in [0,1] ready for a model (or further transform).
'''

def preprocess_image_clahe(image_path, resize=(512, 512)):  # for now, default size is 512x512 (will be used in the model params too)
    # Load
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Resize
    image = cv2.resize(image, resize)

    # Convert BGR → RGB (most research papers and ML frameworks expect RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) # cast to float32 to keep precision for subsequent ops (avoid premature uint8 rounding); float32 avoids loss from integer arithmetic.

    # Step 1: Weighted grayscale conversion
    # computes Igray = 0.2793*R + 0.7041*G + 0.0166*B for every pixel (paper values).
    weights = np.array([0.2793, 0.7041, 0.0166], dtype=np.float32) 
    # tensordot = multiply the 3-channel image by the 3-element weight vector and sum over channels
    image_gray = np.tensordot(image_rgb, weights, axes=([2],[0]))  # resulting shape: (H, W)   ; 2D float array shape

    # Step 2: CLAHE
    '''
    Params:
        > `clipLimit` controls how strongly local histograms are stretched. (SHOULD EXPERIMENT WITH THIS. NEEDS FINETUNING and backup y 5.0 value)
        > `tileGridSize` controls spatial granularity 
    '''
        # 1. divides the image into tiles (8×8)
        # 2. equalizes histograms per tile
        # 3. clips to clipLimit to avoid noise amplification.
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_gray.astype(np.uint8))  # `apply` expects 8-bit images ;  cast to uint8 (0–255).

    # Normalize to [0, 1] for gamma
    image_clahe_norm = image_clahe / 255.0  # many gamma implementations assume normalized inputs ; doing gamma on 0–255 without normalization will give unexpected results

    # Step 3: Gamma correction
       # adjust_gamma applies out = in ** gamma
       # with gamma < 1 (e.g., 0.8), darker regions are brightened, which helps reveal faint vessels
    image_gamma = exposure.adjust_gamma(image_clahe_norm, gamma=0.8)

    # Final output in model format: [C, H, W]
       # expand_dims(..., axis=0) makes the output shape (1, H, W) — a single-channel image ordered as (C, H, W) ; PyTorch convention.
       # astype(np.float32) ensures compat with model input
    image_out = np.expand_dims(image_gamma.astype(np.float32), axis=0)  # 1 channel
    return image_out



'''

'''

def preprocess_image_rgb(image_path, resize=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    image = cv2.resize(image, resize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # C, H, W
    return image


'''
 loads a mask as grayscale, resizes, normalizes to [0,1], and returns shape (1, H, W).
'''

def preprocess_mask(mask_path, resize=(512, 512)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at {mask_path}")
    
    mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=0)  # C, H, W
    return mask
