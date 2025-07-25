import json
import cv2
import sys
from scipy.ndimage import binary_dilation, gaussian_filter, shift
from skimage.morphology import remove_small_objects
from skimage.exposure import match_histograms
from skimage.measure import label, regionprops
from scipy.special import expit
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


def get_centroid(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return None
    return props[0].centroid  # (y, x)


def compute_dose_distribution(gt_mask, spacing_mm, is_lung=True):
    # Step 1: Expand GTV by 3 mm → CTV
    radius_voxels = int(np.round(3 / spacing_mm))  # Assuming isotropic spacing
    expanded = binary_dilation(gt_mask, iterations=radius_voxels)

    # Step 2: Gaussian smoothing
    sigma_mm = 6 if is_lung else 4
    sigma_voxels = sigma_mm / spacing_mm
    dose = gaussian_filter(expanded.astype(np.float32), sigma=sigma_voxels)

    return dose


def compute_d98(dose, mask):
    sorted_vals = np.sort(dose[mask > 0].ravel())
    idx = int(np.ceil(0.02 * len(sorted_vals)))  # top 2% gets underdosed
    return sorted_vals[idx] if len(sorted_vals) > idx else 0.0


def compute_dose_penalty(gt_mask, pred_mask, spacing_mm=1.0, is_lung=True):
    # Step 1: Get original dose distribution from GT mask
    dose = compute_dose_distribution(gt_mask, spacing_mm, is_lung)

    # Step 2: Get centroids
    c_gt = get_centroid(gt_mask)
    c_pred = get_centroid(pred_mask)
    if c_gt is None or c_pred is None:
        return 1.0  # Maximum penalty

    shift_voxels = [c_gt[0] - c_pred[0], c_gt[1] - c_pred[1]]

    # Step 3: Shift the dose distribution
    shifted_dose = shift(dose, shift=shift_voxels, order=1, mode='nearest')

    # Step 4: Compute D98% in GT region for both
    d98_gt = compute_d98(dose, gt_mask)
    d98_shifted = compute_d98(shifted_dose, gt_mask)

    # Step 5: Return relative drop
    if d98_gt == 0:
        return 1.0
    return (d98_gt - d98_shifted) / d98_gt  # relative D98% drop

def ensemble(
    img,
    trackers,
    prev_mask,
    initial_mask,
    w_dice=1.0,
    w_penalty=1.0,
    w_dose=0.5,
    spacing_mm=1.0,
    is_lung=True,
    binarize_thresh=0.5,
    normalize_scores=True
):
    """
    Ensemble tracking predictions using weighted Dice, shape penalty, and dose penalty.

    Parameters:
    - img: current input image
    - trackers: list of tracker models (must implement `.track(img)` and return (pred_mask, logits))
    - prev_mask: binary mask from previous frame
    - initial_mask: binary ground-truth mask from first frame
    - w_dice, w_penalty, w_dose: weighting factors for Dice, shape, and dose penalties
    - spacing_mm: voxel spacing for dose metric
    - is_lung: whether the target is lung tissue (affects dose smoothing)
    - binarize_thresh: threshold for final binary mask
    - normalize_scores: whether to normalize weights to sum to 1

    Returns:
    - final_mask: ensembled binary segmentation mask
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
        dose = compute_dose_penalty(initial_mask, pred_mask, spacing_mm, is_lung)

        score = w_dice * d - w_penalty * p - w_dose * dose

        weights.append(score)
        probs.append(prob)

    weights = np.array(weights)

    # Remove extremely bad predictions (optional)
    weights[weights < 0] = 0

    if normalize_scores and weights.sum() > 0:
        weights = np.exp(weights - np.max(weights))
        weights /= weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    # Weighted average of probability maps
    final_prob = np.tensordot(weights, np.stack(probs, axis=0), axes=1)
    final_mask = (final_prob > binarize_thresh).astype(np.uint8)

    return final_mask


def dice_score(pred, gt, smooth=1e-6):
    intersection = np.logical_and(pred, gt).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return dice