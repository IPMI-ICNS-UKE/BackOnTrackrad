"""
Edit this file to implement your algorithm. 

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
"""
from pathlib import Path
import numpy as np
from PIL import Image

from dam4sam.dam4sam_tracker import DAM4SAMTracker
RESOURCE_PATH = Path("resources")  # load weights and other resources from this directory


def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    tracker = DAM4SAMTracker('sam21pp-L', base_path="/opt/ml/model/")
    initial_mask = target[..., 0].astype(bool)
    frames = ((frames - frames.min()) / (frames.max() - frames.min())) * 255
    frames = frames.astype(np.uint8)
    initial_frame = Image.fromarray(np.repeat(frames[..., 0:1], 3, -1), mode='RGB')
    tracker.initialize(initial_frame, initial_mask)
    tracked_masks = [initial_mask]
    for i in range(1, frames.shape[-1]):
        frame = Image.fromarray(np.repeat(frames[..., i:i+1], 3, -1), mode='RGB')
        tracked_masks.append(tracker.track(frame)['pred_mask'])
    tracked_masks = np.stack(tracked_masks, axis=-1)

    """
    Implement your algorithm here.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
    - target (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
    - frame_rate (float): The frame rate of the MRI linac series.
    - magnetic_field_strength (float): The magnetic field strength of the MRI linac series.
    - scanned_region (str): The scanned region of the MRI linac series.
    """
    
    # frames.shape == (W, H, T)
    # target.shape == (W, H, 1)

    # For the example we want to repeat the initial segmentation for every frame 
    # repeated_target = np.repeat(target, frames.shape[2], axis=-1)

    # repeated_target.shape == (W, H, T)
    return tracked_masks


if __name__ == "__main__":
    run_algorithm(np.random.randn(512,512, 10).astype(np.float32),
                  np.random.randn(512,512,1).astype(np.float32) > 0,
                  0.0,
                  0.0,
                  "head")