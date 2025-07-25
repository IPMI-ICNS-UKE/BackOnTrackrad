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
from skimage.morphology import remove_small_objects
from processing import ensemble, crop_to_bbox, pad_to_full_image, preprocess_stable

RESOURCE_PATH = Path("resources")  # load weights and other resources from this directory


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
            w_penalty=1.0,
            w_dose=1.0,
            is_lung=True if 'lung' in scanned_region else False,
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