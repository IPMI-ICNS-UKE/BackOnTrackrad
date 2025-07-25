"""
Edit this file to implement your algorithm. 

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.append('./dam4sam')
from dam4sam.dam4sam_tracker import DAM4SAMTracker
import cv2
import sys
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import expit
from skimage.morphology import remove_small_objects
import math

RESOURCE_PATH = Path("resources")  # load weights and other resources from this directory


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
    return (255 * image).astype(np.uint8)

ROOT_PATH = Path('/home/tsentker/data/trackrad/trackrad2025_labeled_training_data')
OUT_PATH = Path('/home/tsentker/data/trackrad/out')


def get_mask_stats(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return None
    region = props[0]
    return {
        "area": region.area,
        "centroid": region.centroid,
        "eccentricity": region.eccentricity
    }

def shape_penalty(current_mask, initial_stats, area_tol=0.3, centroid_tol=20, ecc_tol=0.3):
    current_stats = get_mask_stats(current_mask)
    if current_stats is None:
        return float('inf')  # completely reject
    penalty = 0.0

    # Area
    area_diff = abs(current_stats["area"] - initial_stats["area"]) / initial_stats["area"]
    if area_diff > area_tol:
        penalty += area_diff

    # Centroid
    y0, x0 = initial_stats["centroid"]
    y1, x1 = current_stats["centroid"]
    dist = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
    if dist > centroid_tol:
        penalty += dist / 10.0

    # Eccentricity
    ecc_diff = abs(current_stats["eccentricity"] - initial_stats["eccentricity"])
    if ecc_diff > ecc_tol:
        penalty += ecc_diff

    return penalty

# --- Main Wrapper ---

def ensemble(
    img,
    trackers,
    prev_mask,
    initial_mask,
    w_dice=1.0,
    w_penalty=1.0,
    binarize_thresh=0.5,
    normalize_scores=True
):
    """
    trackers: list of tracker models (must have `.track(img)` method returning logits)
    prev_mask: binary mask from previous frame
    initial_mask: binary ground-truth mask from first frame
    Returns: best selected binary mask
    """
    initial_stats = get_mask_stats(initial_mask)
    out = [tracker.track(img) for tracker in trackers]
    pred_masks, logits_list = zip(*out)

    weights = []
    probs = []

    for logits, pred_mask in zip(logits_list, pred_masks):
        prob = expit(logits)

        d = dice_score(pred_mask, prev_mask)
        p = shape_penalty(pred_mask, initial_stats)
        score = w_dice * d - w_penalty * p

        weights.append(score)
        probs.append(prob)

    weights = np.array(weights)

    # Remove extremely bad ones (optional)
    weights[weights < 0] = 0

    if normalize_scores and weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    # Weighted average of probability maps
    final_prob = np.tensordot(weights, np.stack(probs, axis=0), axes=1)

    final_mask = (final_prob > binarize_thresh).astype(np.uint8)
    return final_mask

def ensemble_with_dice_weighting(pred_mask_list, prev_mask, logits_list, use_squared=True, threshold=None):
    """
    logits_list: list of raw logits from different models
    prev_mask: binary reference mask from previous frame
    use_squared: if True, use dice**2 as weights
    threshold: optional min Dice score; if provided, masks below it get weight 0
    """
    dice_scores = [dice_score(pred, prev_mask) for pred in pred_mask_list]

    if use_squared:
        weights = [d**2 for d in dice_scores]
    else:
        weights = dice_scores

    if threshold is not None:
        weights = [w if d >= threshold else 0.0 for w, d in zip(weights, dice_scores)]

    weights = np.array(weights)
    total_weight = np.sum(weights)

    if total_weight == 0:
        print("All predictions below threshold. Using previous mask.")
        return prev_mask.copy()

    # Weighted average of probabilities
    probs = [expit(logits) * w for logits, w in zip(logits_list, weights)]
    mean_prob = np.sum(probs, axis=0) / total_weight
    ensemble_mask = (mean_prob > 0.5)

    return ensemble_mask

def dice_score(pred, gt, smooth=1e-6):
    intersection = np.logical_and(pred, gt).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return dice

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    tracker_t = DAM4SAMTracker('sam21pp-T', base_path=RESOURCE_PATH)
    tracker_s = DAM4SAMTracker('sam21pp-S', base_path=RESOURCE_PATH)
    tracker_b = DAM4SAMTracker('sam21pp-B', base_path=RESOURCE_PATH)
    tracker_l = DAM4SAMTracker('sam21pp-L', base_path=RESOURCE_PATH)

    initial_mask = target[..., 0]
    tracked_masks = [initial_mask]

    initial_frame = preprocess_stable(frames[..., 0:1])
    initial_shape = initial_frame[..., 0].shape
    initial_frame, initial_mask, bbox = crop_to_bbox(image=initial_frame[..., 0], mask=initial_mask, bbox=None, margin=50,
                                             size_multiple=256)

    initial_frame = Image.fromarray(np.repeat(initial_frame[..., None], 3, -1), mode='RGB')

    tracker_t.initialize(initial_frame, initial_mask)
    tracker_s.initialize(initial_frame, initial_mask)
    tracker_b.initialize(initial_frame, initial_mask)
    tracker_l.initialize(initial_frame, initial_mask)

    trackers = [tracker_t, tracker_s, tracker_b, tracker_l]
    for i in range(1, frames.shape[-1]):
        frame = preprocess_stable(frames[..., i:i + 1])
        frame, _, _ = crop_to_bbox(image=frame[..., 0], mask=None, bbox=bbox)
        frame = Image.fromarray(np.repeat(frame[..., None], 3, -1), mode='RGB')

        prev_mask, _, _ = crop_to_bbox(image=tracked_masks[-1], mask=None, bbox=bbox)

        selected_mask = ensemble(
            frame,
            trackers,
            prev_mask,
            initial_mask,
            w_dice=1.0,
            w_penalty=0.5
        )

        selected_mask = pad_to_full_image(selected_mask, initial_shape, bbox)
        tracked_masks.append(selected_mask)

    tracked_masks = np.stack(tracked_masks, axis=-1)

    return tracked_masks


if __name__ == "__main__":
    run_algorithm(np.random.randn(512,512,10).astype(np.float32),
                  np.random.randn(512,512,1).astype(np.float32) > 0,
                  0.0,
                  0.0,
                  "head")