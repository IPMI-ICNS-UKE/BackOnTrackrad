import numpy as np
from dam4sam.dam4sam_tracker import DAM4SAMTracker
from PIL import Image
from tqdm import trange, tqdm
import SimpleITK as sitk
from pathlib import Path
from processing import crop_to_bbox, pad_to_full_image, preprocess_stable, dice_score

ROOT_PATH = Path('/mnt/d/trackrad2025_bad_labeled_training_data')
OUT_PATH = Path('/mnt/d/trackrad_out')
WEIGHT_PATH = Path('/mnt/c/Users/magge/Documents/Code/DAM4SAM/checkpoints')


def _load_mha(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img


def _write_mha(path, img):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)


if __name__ == '__main__':
    tbar = tqdm(list(ROOT_PATH.iterdir()))
    dices = []
    multi_tracker = DAM4SAMTracker(tracker_names=['medsam2pp-T', 'sam21pp-T', 'sam21pp-S', 'sam21pp-B'],# 'sam21pp-L'],
                                   base_path=WEIGHT_PATH)

    for folder in tbar:
        case_name = folder.name
        image_path = (folder / 'images') / f'{case_name}_frames.mha'
        initial_mask_path = (folder / 'targets') / f'{case_name}_first_label.mha'
        targets_path = (folder / 'targets') / f'{case_name}_labels.mha'
        imgs = _load_mha(image_path)
        initial_mask = _load_mha(initial_mask_path)[..., 0]

        out = [initial_mask]
        targets = _load_mha(targets_path)
        img_0 = preprocess_stable(imgs[..., 0:1])
        initial_shape = img_0[..., 0].shape

        img_0, initial_mask, bbox = crop_to_bbox(image=img_0[..., 0],
                                                 mask=initial_mask,
                                                 bbox=None,
                                                 margin=50,
                                                 size_multiple=256)
        initial_mask = initial_mask.astype(np.uint8)

        img_0 = np.repeat(img_0[None], 3, 0)
        multi_tracker.initialize(img_0, initial_mask,
                                 spacing_mm=1.0,
                                 is_lung=True,
                                 ensembling_params=dict(w_dice=1.0,
                                                        w_penalty=1.0,
                                                        w_dose=1.0,
                                                        normalize=True,
                                                        binarize_thresh=0.5))

        for i in trange(1, imgs.shape[-1]):
            img = preprocess_stable(imgs[..., i:i + 1])
            img, _, _ = crop_to_bbox(image=img[..., 0], mask=None, bbox=bbox)
            img = np.repeat(img[None], 3, 0)

            # prev_mask, _, _ = crop_to_bbox(image=out[-1], mask=None, bbox=bbox)

            selected_mask = multi_tracker.track(img)
            selected_mask = pad_to_full_image(selected_mask, initial_shape, bbox)
            out.append(selected_mask)
        pred_mask = np.stack(out, axis=-1)
        dice = dice_score(pred_mask.astype(bool), targets.astype(bool))
        dices.append(dice)
        tbar.set_description(f'mean_dice = {np.mean(dices):.4f}')
        print(f'{case_name}: {dice:.4f}')
        # _write_mha(OUT_PATH/f'__1__{dice}_{case_name}_pred.mha', pred_mask.astype(np.uint8))
        print(f'Dice Score: {np.mean(dices):.4f}')
