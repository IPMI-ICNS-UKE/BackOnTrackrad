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

RESOURCE_PATH = Path("resources")  # load weights and other resources from this directory


def crop_or_pad(
    target_shape,
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    image_pad_value=-1000,
    mask_pad_value=0,
    no_crop: bool = False,
):
    n_dim = len(target_shape)
    if image is None:
        current_shape = mask.shape[-n_dim:]
    else:
        current_shape = image.shape

    pad_width = [(0, 0)] * n_dim
    cropping_slicing = [
        slice(None, None),
    ] * n_dim

    for i_axis in range(n_dim):
        if target_shape[i_axis] is not None:
            if current_shape[i_axis] < target_shape[i_axis]:
                # pad
                padding = target_shape[i_axis] - current_shape[i_axis]
                padding_left = padding // 2
                padding_right = padding - padding_left
                pad_width[i_axis] = (padding_left, padding_right)

            elif not no_crop and current_shape[i_axis] > target_shape[i_axis]:
                # crop
                cropping = current_shape[i_axis] - target_shape[i_axis]
                cropping_left = cropping // 2
                cropping_right = cropping - cropping_left

                cropping_slicing[i_axis] = slice(cropping_left, -cropping_right)

    if image is not None:
        image = np.pad(
            image,
            pad_width,
            mode="constant",
            constant_values=image_pad_value,
        )
    if mask is not None:
        extra_dims = mask.ndim - n_dim
        mask = np.pad(
            mask,
            [(0, 0)] * extra_dims + pad_width,
            mode="constant",
            constant_values=mask_pad_value,
        )

    cropping_slicing = tuple(cropping_slicing)
    if image is not None:
        image = image[cropping_slicing]
    if mask is not None:
        mask_cropping_slicing = cropping_slicing
        if mask.ndim > n_dim:
            mask_cropping_slicing = (..., *cropping_slicing)
        mask = mask[mask_cropping_slicing]

    return image, mask

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

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    tracker_t = DAM4SAMTracker('sam21pp-T', base_path=RESOURCE_PATH)
    tracker_s = DAM4SAMTracker('sam21pp-S', base_path=RESOURCE_PATH)
    tracker_b = DAM4SAMTracker('sam21pp-B', base_path=RESOURCE_PATH)
    tracker_l = DAM4SAMTracker('sam21pp-L', base_path=RESOURCE_PATH)

    initial_mask = target[..., 0].astype(bool)
    tracked_masks = [initial_mask]

    initial_frame = preprocess_stable(frames[..., 0:1])
    initial_shape = initial_frame[..., 0].shape
    initial_frame, initial_mask = crop_or_pad(target_shape=(256, 256), image=initial_frame[..., 0], mask=initial_mask)
    initial_frame = Image.fromarray(np.repeat(initial_frame[..., None], 3, -1), mode='RGB')

    tracker_t.initialize(initial_frame, initial_mask)
    tracker_s.initialize(initial_frame, initial_mask)
    tracker_b.initialize(initial_frame, initial_mask)
    tracker_l.initialize(initial_frame, initial_mask)

    for i in range(1, frames.shape[-1]):
        frame = preprocess_stable(frames[..., i:i + 1])
        frame, _ = crop_or_pad(target_shape=(256, 256), image=frame[..., 0])
        frame = Image.fromarray(np.repeat(frame[..., None], 3, -1), mode='RGB')

        _, logits_t = tracker_t.track(frame)
        _, logits_s = tracker_s.track(frame)
        _, logits_b = tracker_b.track(frame)
        _, logits_l = tracker_l.track(frame)

        logits = np.stack([logits_t, logits_s, logits_b, logits_l], axis=-1)
        logits = np.mean(expit(logits), axis=-1)
        logits = (logits > 0.5).astype(np.uint8)
        logits = remove_small_objects(logits, min_size=28).astype(np.uint8)
        _, logits = crop_or_pad(target_shape=initial_shape, image=None, mask=logits)

        tracked_masks.append(logits)

    tracked_masks = np.stack(tracked_masks, axis=-1)

    return tracked_masks


if __name__ == "__main__":
    run_algorithm(np.random.randn(512,512,10).astype(np.float32),
                  np.random.randn(512,512,1).astype(np.float32) > 0,
                  0.0,
                  0.0,
                  "head")