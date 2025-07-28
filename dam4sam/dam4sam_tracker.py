from copy import deepcopy

import numpy as np
import yaml
from hydra import compose
import torch
import torchvision.transforms.functional as F
from scipy.special import expit
from scipy.ndimage import center_of_mass

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType

from processing import dice_score, get_mask_stats, shape_penalty, center_of_mass_penalty, shifted_dice
from sam2.build_sam import build_sam2_video_predictor
from collections import OrderedDict
import random
import os
from dam4sam.utils.utils import keep_largest_component, determine_tracker

from pathlib import Path
# config_path = Path(__file__).parent / "dam4sam_config.yaml"
# with open(config_path) as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
#
seed = 0
# seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class DAM4SAMTracker():
    def __init__(self, tracker_names=["sam21pp-L"], base_path=None):
        """
        Constructor for the DAM4SAM (2 or 2.1) tracking wrapper.

        Args:
        - tracker_name (str): Name of the tracker to use. Options are:
            - "sam21pp-L": DAM4SAM (2.1) Hiera Large
            - "sam21pp-B": DAM4SAM (2.1) Hiera Base+
            - "sam21pp-S": DAM4SAM (2.1) Hiera Small
            - "sam21pp-T": DAM4SAM (2.1) Hiera Tiny
            - "sam2pp-L": DAM4SAM (2) Hiera Large
            - "sam2pp-B": DAM4SAM (2) Hiera Base+
            - "sam2pp-S": DAM4SAM (2) Hiera Small
            - "sam2pp-T": DAM4SAM (2) Hier Tiny
        """
        self.predictors = []
        for t in tracker_names:
            self.checkpoint, self.model_cfg = determine_tracker(t, base_path)
            model_cfg = compose(config_name=self.model_cfg, overrides=[
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
        ])
            # Image preprocessing parameters
            self.input_image_size = model_cfg.model.image_size
            self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
            self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

            self.predictors.append(build_sam2_video_predictor(self.model_cfg, self.checkpoint, device="cuda:0"))
        self.tracking_times = []

    def _prepare_image(self, img):
        # _load_img_as_tensor from SAM2
        img = torch.from_numpy(img).to(self.inference_state["device"])
        # img = img.permute(2, 0, 1).float()# / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean) / self.img_std        
        return img

    @torch.inference_mode()
    def init_state_tw(
        self,
    ):
        """Initialize an inference state."""
        compute_device = torch.device("cuda")
        inference_state = {}
        inference_state["images"] = None # later add, step by step
        inference_state["num_frames"] = 0 # later add, step by step
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = False
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = False
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = None # later add, step by step
        inference_state["video_width"] =  None # later add, step by step
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device #torch.device("cpu")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["adds_in_drm_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        
        self.img_mean = self.img_mean.to(compute_device)
        self.img_std = self.img_std.to(compute_device)

        return inference_state
    
    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None, ensembling_params=None):
        """
        Initialize the tracker with the first frame and mask.
        Function builds the DAM4SAM (2.1) tracker and initializes it with the first frame and mask.

        Args:
        - image (PIL Image): First frame of the video.
        - init_mask (numpy array): Binary mask for the initialization
        
        Returns:
        - out_dict (dict): Dictionary containing the mask for the initialization frame.
        """
        if type(init_mask) is list:
            init_mask = init_mask[0]
        self.initial_stats = get_mask_stats(init_mask)
        self.init_mask = init_mask
        self.ensembling_params = ensembling_params
        self.prev_mask = init_mask
        self.prev_centroids = []
        self.frame_index = 0 # Current frame index, updated frame-by-frame
        self.object_sizes = [] # List to store object sizes (number of pixels)
        self.all_masks = []
        self.last_added = -1 # Frame index of the last added frame into DRM memory
        
        self.img_width = image.shape[-1]  # Original image width
        self.img_height = image.shape[-2]  # Original image height
        self.inference_state = self.init_state_tw()
        self.inference_state["images"] = image
        video_width, video_height = image.shape[-1], image.shape[-2]
        self.inference_state["video_height"] = video_height
        self.inference_state["video_width"] =  video_width
        prepared_img = self._prepare_image(image)
        self.inference_state["images"] = {0 : prepared_img}
        self.inference_state["num_frames"] = 1
        self.inference_states = [deepcopy(self.inference_state) for i in range(len(self.predictors))]
        [p.reset_state(i) for i, p in zip(self.inference_states, self.predictors)]

        # warm up the model
        [p._get_image_feature(i, frame_idx=0, batch_size=1) for i, p in zip(self.inference_states,self.predictors)]

        if init_mask is None:
            if bbox is None:
                print('Error: initialization state (bbox or mask) is not given.')
                exit(-1)
            
            # consider bbox initialization - estimate init mask from bbox first
            init_mask = self.estimate_mask_from_box(bbox)

        out_mask_logits = []
        masks = []
        for i, p in zip(self.inference_states, self.predictors):
            _, _, out_mask_logit = p.add_new_mask(
                inference_state=i,
                frame_idx=0,
                obj_id=0,
                mask=init_mask,
            )
            out_mask_logit = out_mask_logit[0][0].float().cpu().numpy() # Extract the mask logits
            out_mask_logits.append(out_mask_logit)
            masks.append(out_mask_logit > 0)
        # TODO combine inference states
        m = masks[0]
        [i["images"].pop(self.frame_index) for i in self.inference_states]  # Remove the image from the inference state

        out_dict = {'pred_mask': m}
        return out_dict

    def _ensemble_masks(self, out_mask_logits,
                        pred_masks,
                        normalize=True,
                        threshold_list=[0.45, 0.5, 0.66],
                        w_dice=1.0,
                        frame_rate=1,
                        w_penalty=1.0,
                        w_com=1.0):
        weights = []
        probs = []
        self.prev_centroids.append(center_of_mass(self.prev_mask))

        for logits, pred_mask in zip(out_mask_logits, pred_masks):
            prob = expit(logits)
            d = shifted_dice(pred_mask, self.prev_mask, self.prev_centroids, frame_rate)
            p = shape_penalty(pred_mask, self.prev_mask)
            com = center_of_mass_penalty(pred_mask, self.prev_centroids, frame_rate)
            score = w_dice * d - w_penalty * p - w_com * com

            weights.append(score)
            probs.append(prob)

        weights = np.array(weights)

        # Remove extremely bad predictions (optional)
        weights[weights < 0] = 0

        if normalize and weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        # Weighted average of probability maps
        final_prob = np.tensordot(weights, np.stack(probs, axis=0), axes=1)
        best_score = None
        best_mask = None
        for thresh in threshold_list:
            mask = (final_prob > thresh).astype(np.uint8)
            d = shifted_dice(mask, self.prev_mask, self.prev_centroids, frame_rate)
            p = shape_penalty(mask, self.prev_mask)
            com = center_of_mass_penalty(mask, self.prev_centroids, frame_rate)
            score = w_dice * d - w_penalty * p - w_com * com
            if best_score is None or score > best_score:
            # if score > best_score:
                best_score = score
                best_mask = mask

        return best_mask

    def compute_iou(self, mask1, mask2):
        """
        Compute Intersection over Union (IoU) between two binary masks.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 1.0  # Both masks are empty (no tumor)
        return intersection / union

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        Function to track the object in the next frame.

        Args:
        - image (numpy array): Next frame of the video.
        - init (bool): Whether the current frame is the initialization frame.

        Returns:
        - out_dict (dict): Dictionary containing the predicted mask for the current frame.
        """
        # TODO autocast?!
        torch.cuda.empty_cache()
        # Prepare the image for input to the model
        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            for i in self.inference_states:
                i["num_frames"] += 1
        # Propagate the tracking to the next frame
        # return_all_masks=True returns all predicted (chosen and alternative) masks and corresponding IoUs
        out_mask_logits = []
        masks = []
        ious = []
        for i, p in zip(self.inference_states, self.predictors):
            i["images"][self.frame_index] = prepared_img
            # TODO propagate in video allways uses all previous state this must be possible more efficiently
            out = p.propagate_in_video(
                inference_state=i,
                start_frame_idx=self.frame_index,
                max_frame_num_to_track=0,
                return_all_masks=True
            ).__next__()
            out_mask_logit = out[2]
            out_frame_idx = out[0]
            ious.append(out[3][1].max())
            out_mask_logit = out_mask_logit[0][0].float().cpu().numpy() # Extract the mask logits
            out_mask_logits.append(out_mask_logit)
            masks.append(out_mask_logit > 0)
        m = self._ensemble_masks(out_mask_logits, masks, **self.ensembling_params)
        for i, p in zip(self.inference_states, self.predictors):
            p.add_new_mask(
            inference_state=i,
            frame_idx=self.frame_index,
            obj_id=0,
            mask=m,
            )
        # TODO woher kommt dieser wert eigentlich? können wir dafür irgendwo zwischen einen dice berechnen?  evtl prev frame?
        m_iou = np.mean(ious)

        n_pixels = (m == 1).sum()
        self.object_sizes.append(n_pixels)
        if len(self.object_sizes) > 1 and n_pixels >= 1:
            obj_sizes_ratio = n_pixels / np.median([
                size for size in self.object_sizes[-10:] if size >= 1
            ][-10:])
        else:
            obj_sizes_ratio = -1

        # The first condition checks if:
        #  - the chosen predicted mask has a high predicted IoU,
        #  - the object size ratio is within a +- 20% range compared to the previous frames,
        #  - the target is present in the current frame,
        #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
        # Numpy array of the chosen mask and corresponding bounding box
        # self.all_masks.append(m)
        # n_pixels_prev = (prev_mask == 1).sum()
        # print(f'{n_pixels_prev=}; {n_pixels=}')
        # if (n_pixels / n_pixels_prev) < 0.8 or (n_pixels / n_pixels_prev) > 1.2:
        #     print('Using previous mask for current frame.')
        #     m = prev_mask
        if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and n_pixels >= 1 and (
                self.frame_index - self.last_added > 5 or self.last_added == -1):
            alternative_masks = [
                Mask(m_).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(
                    np.uint8)
                for m_ in masks]

            # Numpy array of the chosen mask and corresponding bounding box
            chosen_mask_np = m.copy()
            chosen_bbox = Mask(chosen_mask_np).convert(RegionType.RECTANGLE)

            # Delete the parts of the alternative masks that overlap with the chosen mask
            alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in
                                 alternative_masks]
            # Keep only the largest connected component of the processed alternative masks
            alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
            if len(alternative_masks) > 0:
                # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                alternative_bboxes = [Mask(m_).convert(RegionType.RECTANGLE) for m_ in alternative_masks]
                # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                ious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]

                # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                # That would mean that there are significant differences between the chosen mask and the processed alternative masks,
                # leading to possible detections of distractors within alternative masks.
                if np.min(np.array(ious)) <= 0.7:
                    self.last_added = self.frame_index  # Update the last added frame index
                    for p in self.predictors:
                        p.add_to_drm(
                            inference_state=self.inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=0,
                        )
        # if m_iou > 0.8 and obj_sizes_ratio >= 0.9 and obj_sizes_ratio <= 1.1 and n_pixels >= 1 and (self.frame_index - self.last_added > 5 or self.last_added == -1):
        #     masks = [Mask(m_).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8)
        #                      for m_ in masks]
        #
        #     chosen_mask_np = m.copy()
        #     # chosen_bbox = Mask(chosen_mask_np).convert(RegionType.RECTANGLE)
        #
        #     # Delete the parts of the alternative masks that overlap with the chosen mask
        #     masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in masks]
        #     # Keep only the largest connected component of the processed alternative masks
        #     masks = [keep_largest_component(m_) for m_ in masks if np.sum(m_) >= 1]
        #     if len(masks) > 0:
        #         # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
        #         masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in masks]
        #         out_all_dices = [dice_score(m, mask) for mask in masks]
        #
        #         # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
        #         # alternative_bboxes = [Mask(m_).convert(RegionType.RECTANGLE) for m_ in alternative_masks]
        #         # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
        #         # ious = [calculate_overlaps([chosen_mask_np], [mask])[0] for mask in alternative_masks]
        #
        #         # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
        #         # That would mean that there are significant differences between the chosen mask and the processed alternative masks,
        #         # leading to possible detections of distractors within alternative masks.
        #         if np.min(np.array(out_all_dices)) <= 0.8:
        #             self.last_added = self.frame_index # Update the last added frame index
        #
        #             [p.add_to_drm(inference_state=i, frame_idx=out_frame_idx, obj_id=0,)
        #              for i, p in zip(self.inference_states, self.predictors)]
        #     # Return the predicted mask for the current frame
        #     # out_dict = {'pred_mask': m}

        self.prev_mask = m.copy()  # Update the previous mask for the next frame
        # TODO update the inference state with the current mask
        [i["images"].pop(self.frame_index) for i in self.inference_states]  # Remove the image from the inference state
        return m.astype(np.uint8)

    def estimate_mask_from_box(self, bbox):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._get_image_feature(self.inference_state, 0, 1)

        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
        box = torch.as_tensor(box, dtype=torch.float, device=current_vision_feats[0].device)

        from sam2.utils.transforms import SAM2Transforms
        _transforms = SAM2Transforms(
            resolution=self.predictor.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        unnorm_box = _transforms.transform_boxes(
            box, normalize=True, orig_hw=(self.img_height, self.img_width)
        )  # Bx2x2
        
        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
        box_labels = box_labels.repeat(unnorm_box.size(0), 1)
        concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(current_vision_feats[i].permute(1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))
        if self.predictor.directly_add_no_mem_embed:
            img_embed = current_vision_feats[2] + self.predictor.no_mem_embed
        else:
            img_embed = current_vision_feats[2]
        _, b_, c_ = current_vision_feats[2].shape
        img_embed = img_embed.permute(1, 2, 0).view(b_, c_, feat_sizes[2][0], feat_sizes[2][1])
        low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = _transforms.postprocess_masks(
            low_res_masks, (self.img_height, self.img_width)
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        masks = masks > 0

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()

        init_mask = masks_np[0]
        return init_mask
