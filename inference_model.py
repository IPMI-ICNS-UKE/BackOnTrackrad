"""
Edit this file to implement your algorithm. 

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
"""
from pathlib import Path
from scipy.ndimage import binary_closing

import sys
sys.path.append('./dam4sam')
from dam4sam.dam4sam_tracker import DAM4SAMTracker
from skimage.exposure import match_histograms
import math
import numpy as np


def pad_to_full_image(cropped, full_shape, bbox):
    """
    Place a cropped image or mask back into the original full-size image/mask.

    Parameters:
        cropped (ndarray): Cropped 2D or 3D image/mask.
        full_shape (tuple): Shape of the full-size image, e.g. (H, W) or (H, W, C).
        bbox (tuple): (y1, y2, x1, x2) coordinates used during cropping.

    Returns:
        padded (ndarray): Full-size image with the cropped content inserted.
    """
    y1, y2, x1, x2 = bbox

    # Sanity check
    if cropped.shape[:2] != (y2 - y1, x2 - x1):
        raise ValueError(
            f"Shape mismatch: cropped shape {cropped.shape[:2]} does not match bbox {(y2 - y1, x2 - x1)}")

    # Initialize full-size array
    padded = np.zeros(full_shape, dtype=cropped.dtype)

    if cropped.ndim == 2 or (cropped.ndim == 3 and len(full_shape) == 2):
        padded[y1:y2, x1:x2] = cropped
    else:
        padded[y1:y2, x1:x2, :] = cropped

    return padded

def crop_to_bbox(image, mask=None, bbox=None, margin=10, size_multiple=64):
    """
    Crop a square region from the image (and optionally mask), where the side length
    is the smallest multiple of `size_multiple` that contains the tumor (from mask or bbox).

    If a mask is provided and bbox is None:
        - Compute square bbox from mask with margin.
        - Enlarge to nearest multiple of size_multiple.

    If bbox is provided:
        - Use as-is (no margin, no resizing to multiple).

    Parameters:
        image (ndarray): 2D or 3D image (H, W) or (H, W, C).
        mask (ndarray, optional): Binary mask for computing bounding box.
        bbox (tuple, optional): (y1, y2, x1, x2) to crop directly.
        margin (int): Extra pixels around the mask (only used if mask is provided).
        size_multiple (int): Ensure output size is a multiple of this value (default 64).

    Returns:
        cropped_image (ndarray)
        cropped_mask (ndarray or None)
        final_bbox (tuple): (y1, y2, x1, x2)
    """
    H, W = image.shape[:2]

    if bbox is not None:
        # Use provided bbox directly
        y1, y2, x1, x2 = bbox
        if not (0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W):
            raise ValueError(f"Invalid bbox {bbox} for image shape {image.shape}")
    else:
        if mask is None or not np.any(mask):
            raise ValueError("Must provide non-empty mask if bbox is not given.")

        y_coords, x_coords = np.where(mask > 0)
        y_min = y_coords.min()
        y_max = y_coords.max()
        x_min = x_coords.min()
        x_max = x_coords.max()

        # Apply margin
        y1 = max(0, y_min - margin)
        y2 = min(H, y_max + margin + 1)
        x1 = max(0, x_min - margin)
        x2 = min(W, x_max + margin + 1)

        # Compute square center
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2

        # Compute minimal square size needed
        height = y2 - y1
        width = x2 - x1
        box_size = max(height, width)

        # Round up to next multiple of size_multiple
        box_size = int(math.ceil(box_size / size_multiple) * size_multiple)

        # Compute final square bbox centered at (cy, cx)
        half = box_size // 2
        y1 = max(0, cy - half)
        y2 = y1 + box_size
        if y2 > H:
            y2 = H
            y1 = H - box_size

        x1 = max(0, cx - half)
        x2 = x1 + box_size
        if x2 > W:
            x2 = W
            x1 = W - box_size

        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(H, y2)
        x2 = min(W, x2)

    # Crop
    cropped_image = image[y1:y2, x1:x2] if image.ndim == 2 else image[y1:y2, x1:x2, :]
    cropped_mask = mask[y1:y2, x1:x2] if mask is not None else None

    final_bbox = (y1, y2, x1, x2)
    return cropped_image, cropped_mask, final_bbox

def close_mask(mask, structure=None):
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)  # or (5,5)
    return binary_closing(mask, structure=structure).astype(np.uint8)

def preprocess_stable(image, reference=None):
    """
    MRI preprocessing that is tracker/model safe.
    If reference is provided, match histogram to it.
    """
    # image = (image - np.percentile(image, 1)) / (np.percentile(image, 99) - np.percentile(image, 1) + 1e-8)
    image = (image - image.min()) / (image.max() - image.min())
    # image = np.clip(image, 0, 1)

    # 3. Match histogram to first frame (optional, only if reference is available)
    if reference is not None:
        reference = reference / 255
        image = match_histograms(image, reference, channel_axis=None)

    # 4. Convert to uint8 for downstream
    return image


RESOURCE_PATH = Path("/opt/ml/models")  # load weights and other resources from this directory

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    multi_tracker = DAM4SAMTracker(tracker_names=['sam21pp-T', 'sam21pp-S', 'sam21pp-B'],#, 'sam21pp-L'],
                                   base_path=RESOURCE_PATH)

    initial_mask = target[..., 0]
    tracked_masks = [initial_mask]
    initial_shape = frames.shape[:2]

    initial_frame, initial_mask, bbox = crop_to_bbox(image=frames[..., 0], mask=initial_mask, bbox=None, margin=50,
                                             size_multiple=256)
    initial_frame = preprocess_stable(initial_frame)

    initial_frame = np.repeat(initial_frame[None], 3, 0)

    multi_tracker.initialize(initial_frame, initial_mask, ensembling_params=dict(normalize=True, frame_rate=frame_rate))

    for i in range(1, frames.shape[-1]):
        frame, _, _ = crop_to_bbox(image=frames[..., i], mask=None, bbox=bbox)
        frame = preprocess_stable(frame)
        frame = np.repeat(frame[None], 3, 0)

        selected_mask = multi_tracker.track(frame)
        selected_mask = pad_to_full_image(selected_mask, initial_shape, bbox)
        selected_mask = close_mask(selected_mask.astype(bool))

        tracked_masks.append(selected_mask)

    tracked_masks = np.stack(tracked_masks, axis=-1)

    return tracked_masks


if __name__ == "__main__":
    run_algorithm(np.random.randn(512,512,10).astype(np.float32),
                  np.random.randn(512,512,1).astype(np.float32) > 0,
                  0.0,
                  0.0,
                  "head")