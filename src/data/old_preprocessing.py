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
        # gray_pixel = R * w_R + G * w_G + B * w_B
    image_gray = np.tensordot(image_rgb, weights, axes=([2],[0]))  # resulting shape: (H, W)   ; 2D float array shape
    '''
    ex.
    image_rgb = (H,W,3) 
        [
        [[100, 150, 200], [ 50,  80, 120]],
        [[255, 255, 255], [  0,   0,   0]]
        ]
    weights = [0.3, 0.6, 0.1]


    image_gray = (H,W)
        [
        [140.0, 77.0],
        [255.0, 0.0]
        ]
    '''    




# Step 2: CLAHE
    '''
    Params:
        > `clipLimit` controls how strongly local histograms are stretched.
        > `tileGridSize` controls spatial granularity 

        should fine tune values when training
    '''
        # 1. divides the image into tiles (8×8)
        # 2. equalizes histograms per tile
        # 3. clips to clipLimit to avoid noise amplification.
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))  # NOTE!!! i changed this to 16,16 from 8,8 -- might need to adjust in paper and compare in model later
    image_clahe = clahe.apply(image_gray.astype(np.uint8))  # `apply` expects 8-bit images ;  cast to uint8 (0–255).

# Normalize to [0, 1] for gamma
    image_clahe_norm = image_clahe / 255.0  # many gamma implementations assume normalized inputs ; doing gamma on 0–255 without normalization will give unexpected results

# Step 3: Gamma correction
       # adjust_gamma applies out = in ** gamma
       # with gamma < 1 (e.g., 0.8), darker regions are brightened, which helps reveal faint vessels
    image_gamma = exposure.adjust_gamma(image_clahe_norm, gamma=0.75) # NOTE!!! i also changed this from 8 to 0.75

# Final output in model format: [C, H, W]
       # expand_dims(..., axis=0) makes the output shape (1, H, W) — a single-channel image ordered as (C, H, W) ; PyTorch convention.
       # astype(np.float32) ensures compat with model input
    image_out = np.expand_dims(image_gamma.astype(np.float32), axis=0)  # 1 channel
    return image_out



'''
reads an RGB image from a given path, resizes it, normalizes pixel values to [0,1],
and converts it to channel-first format (C, H, W) for deep learning models.
'''
def preprocess_image_rgb(image_path, resize=(512, 512)):
# Load image from file (BGR format by default in OpenCV)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
# Resize image to the target dimensions
    image = cv2.resize(image, resize)
    
# Convert from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
# Rearrange dimensions from (H, W, C) → (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    return image


'''
Loads a segmentation mask as grayscale, resizes it, normalizes pixel values to [0,1],
and adds a channel dimension so the shape is (1, H, W).
'''
def preprocess_mask(mask_path, resize=(512, 512)):
# Load mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at {mask_path}")
    
# Resize mask with nearest-neighbor interpolation (to preserve class labels)
    mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
    
# Normalize pixel values to range [0,1]
    mask = mask.astype(np.float32) / 255.0
    
# Add channel dimension (C=1)
    mask = np.expand_dims(mask, axis=0)
    
    return mask
