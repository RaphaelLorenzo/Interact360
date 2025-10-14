import torch
import os
import argparse
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import socket
import pickle
from typing import Optional, Union
from ultralytics import YOLO

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from sam2.build_sam import build_sam2_camera_predictor, build_sam2
from sam2.sam2_camera_predictor import SAM2CameraPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from transformers import AutoProcessor, VitPoseForPoseEstimation

from misc.print_utils import *
from misc.other_utils import *
from misc.rle_tools import encode_RLE, decode_RLE

import warnings
# filter UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

SAM2_MODEL_TO_CHECKPOINT = {
    "tiny": os.path.join(here, "../", "checkpoints/sam2.1_hiera_tiny.pt"),
    "tiny_512": os.path.join(here, "../", "checkpoints/sam2.1_hiera_tiny.pt"),
    "small": os.path.join(here, "../", "checkpoints/sam2.1_hiera_small.pt"),
    "base": os.path.join(here, "../", "checkpoints/sam2.1_hiera_base_plus.pt"),
    "large": os.path.join(here, "../", "checkpoints/sam2.1_hiera_large.pt")
}

SAM2_MODEL_TO_CFG = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "tiny_512": "configs/sam2.1/sam2.1_hiera_t_512.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml", 
    "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml"
}

ERODE_AND_DILATE_N_PIXELS = 11
MIN_BOX_SIZE_WIDTH_PROP_OF_CENTERED_IMAGE = 0.01
MIN_BOX_SIZE_HEIGHT_PROP_OF_CENTERED_IMAGE = 0.05
MIN_MASK_PIXELS_PROP_OF_CENTERED_IMAGE = 0.002 # 0.2% of the image (~7K pixels for 1920x1920)
MIN_VITPOSE_VALID = 11
MIN_VITPOSE_VALID_IN_MASK = 7
MAX_OCCLUSION_FRAMES_COUNT = 30
MIN_OVERLAP_IOU_BOXES = 0.5
MIN_OVERLAP_IOU_MASKS = 0.7
MIN_MASK_IN_MASK_COMP_PROP = 0.95 # if 95% of the current mask is in the compared mask, we consider that we actually are a subpart of it and break

COLORS_SEQUENCE_BGR = [(98, 170, 174), (200, 150, 168), (213, 208, 44), (191, 14, 148), (150, 138, 149), 
            (133, 161, 102), (105, 161, 43), (71, 120, 13), (13, 81, 212), (44, 150, 224), 
            (243, 207, 71), (157, 248, 179), (121, 164, 71), (34, 93, 202), (175, 49, 4), 
            (43, 248, 163), (38, 20, 240), (59, 191, 20), (142, 8, 82), (219, 88, 31), 
            (37, 147, 24), (83, 137, 140), (218, 35, 175), (94, 237, 138), (35, 91, 70), 
            (7, 55, 87), (223, 77, 210), (173, 146, 250), (1, 131, 67), (62, 12, 215)]  

def get_iou_torch(track_mask, local_mask):
    assert(track_mask.dtype == torch.bool), "Track mask must be a boolean tensor got {}".format(track_mask.dtype)
    assert(local_mask.dtype == torch.bool), "Local mask must be a boolean tensor got {}".format(local_mask.dtype)
    assert(track_mask.shape == local_mask.shape), "Track mask and local mask must have the same shape got {} and {}".format(track_mask.shape, local_mask.shape)
    assert(track_mask.device == local_mask.device), "Track mask and local mask must be on the same device got {} and {}".format(track_mask.device, local_mask.device)
    # compute the iou
    iou = torch.sum(track_mask & local_mask) / torch.sum(track_mask | local_mask)
    return iou

def prop_of_mask1_in_mask2_torch(mask1, mask2):
    assert(mask1.dtype == torch.bool), "Mask1 must be a boolean tensor got {}".format(mask1.dtype)
    assert(mask2.dtype == torch.bool), "Mask2 must be a boolean tensor got {}".format(mask2.dtype)
    assert(mask1.shape == mask2.shape), "Mask1 and Mask2 must have the same shape got {} and {}".format(mask1.shape, mask2.shape)
    assert(mask1.device == mask2.device), "Mask1 and Mask2 must be on the same device got {} and {}".format(mask1.device, mask2.device)
    # compute the proportion of mask1 in mask2
    prop = torch.sum(mask1 & mask2) / torch.sum(mask1)
    return prop

def bb_intersection_over_union_with_shift(boxA, boxB, shift_test):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes in a 360 image, test two hypothesis each time : 
    - boxA vs boxB
    - boxA vs boxB + shift_test
    - boxA vs boxB - shift_test
    
    And return the best IoU between the 3 hypotheses.

    Args:
        boxA (list): List containing the coordinates of the first bounding box in the format [xmin, ymin, xmax, ymax].
        boxB (list): List containing the coordinates of the second bounding box in the format [xmin, ymin, xmax, ymax].
        shift_test (int): Number of pixels to shift the second box.

    Returns:
        float: The best IoU between the two hypotheses.
    """
    boxB_shifted_plus = boxB.copy()
    boxB_shifted_plus[0] = boxB[0] + shift_test
    boxB_shifted_plus[2] = boxB[2] + shift_test
    boxB_shifted_minus = boxB.copy()
    boxB_shifted_minus[0] = boxB[0] - shift_test
    boxB_shifted_minus[2] = boxB[2] - shift_test
    return max(bb_intersection_over_union(boxA, boxB), bb_intersection_over_union(boxA, boxB_shifted_plus), bb_intersection_over_union(boxA, boxB_shifted_minus))

def get_convex_hull_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Get the convex hull of a mask.

    Args:
        mask (torch.Tensor): Mask to get the convex hull of, must be a boolean tensor. Shape is (H,W)

    Returns:
        torch.Tensor: Convex hull mask, same shape and device as input mask, dtype is torch.bool
    """
    assert mask.dtype == torch.bool, f"Mask must be a boolean tensor got {mask.dtype}"
    assert mask.ndim == 2, f"Mask must be a 2D tensor got {mask.ndim}"

    # import numpy as np
    # import cv2

    # Convert mask to numpy uint8
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or len(contours) == 0:
        # No foreground, return all False
        return torch.zeros_like(mask)

    # Concatenate all points from all contours
    all_points = np.concatenate(contours, axis=0)

    # Compute convex hull
    hull = cv2.convexHull(all_points)

    # Create empty mask
    hull_mask_np = np.zeros_like(mask_np, dtype=np.uint8)

    # Fill convex hull
    cv2.fillConvexPoly(hull_mask_np, hull, 1)

    # Convert back to torch.bool and to original device
    hull_mask = torch.from_numpy(hull_mask_np.astype(bool)).to(mask.device)

    return hull_mask


def bb_intersection_over_union(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list): List containing the coordinates of the first bounding box in the format [xmin, ymin, xmax, ymax].
        boxB (list): List containing the coordinates of the second bounding box in the format [xmin, ymin, xmax, ymax].

    Returns:
        float: The IoU value between the two bounding boxes.
    """
    
    if boxA is None or boxB is None:
        return 0.0
    
    if boxA[0] is None or boxA[1] is None or boxA[2] is None or boxA[3] is None:
        return 0.0
    
    if boxB[0] is None or boxB[1] is None or boxB[2] is None or boxB[3] is None:
        return 0.0
        
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_image_centered_on_detection(image : np.ndarray, detection_box : list, projection_type : str = "horizontal_crop", desired_width: Optional[int] = None) -> tuple:
    """ Get the centered image on a detection box.
    
    Args:
        image (np.ndarray): Image to crop as an ndarray, shape is [H, W, 3]
        detection_box (list): List containing the coordinates of the detection box in the format [xmin, ymin, xmax, ymax].
        projection_type (str): Type of projection to use, either "horizontal_crop" or "perspective".
        desired_width (Optional[int]): Desired width of the centered image. If None, the width of the image is used.
    Returns:
        tuple: Tuple containing the centered image, the centered box, the xmin and xmax of the image in the original image
    """
    
    assert(image.ndim == 3), "Image is expected to be a 3D ndarray got {}".format(image.ndim)
    
    if projection_type == "horizontal_crop":
        
        xmin = detection_box[0]
        ymin = detection_box[1]
        xmax = detection_box[2]
        ymax = detection_box[3]
        center_box_x = (xmin + xmax) / 2
        if desired_width is None:
            desired_width = image.shape[1] // 2
        
        image_xmin = int(center_box_x) - desired_width // 2 # 3858 (3693) |
        image_xmax = int(center_box_x) + desired_width // 2 # 5778 (5613)
        
        if image_xmin >= 0 and image_xmax < image.shape[1]:
            # normal case
            centered_image = image[:, image_xmin:image_xmax, :]
            centered_xmin = xmin - image_xmin
            centered_xmax = xmax - image_xmin
            centered_ymin = ymin
            centered_ymax = ymax
                    
        elif image_xmin < 0 and image_xmax >= 0:
            # detection on the edge (left)
            centered_image = np.concatenate([image[:, image_xmin:, :], image[:, :image_xmax, :]], axis=1)
            centered_xmin = xmin - image_xmin
            centered_xmax = xmax - image_xmin
            centered_ymin = ymin
            centered_ymax = ymax
        
        elif image_xmax >= image.shape[1] and image_xmin < image.shape[1]:
            # detection on the edge (right)
            centered_image = np.concatenate([image[:, image_xmin:, :], image[:, :image_xmax-image.shape[1], :]], axis=1) # 3840-3858=0 + 5778-3840=1938 ==> 1938 (3840-3693=147 and 5613-3840=1773 => 1920)
            centered_xmin = xmin - image_xmin
            centered_xmax = xmax - image_xmin
            centered_ymin = ymin
            centered_ymax = ymax
        
        elif image_xmin < 0 and image_xmax < 0:
            # detection fully on the left
            image_xmin = image_xmin + image.shape[1]
            image_xmax = image_xmax + image.shape[1]
            centered_image = image[:, image_xmin:image_xmax, :]
            
            centered_xmin = xmin + image.shape[1] - image_xmin
            centered_xmax = xmax + image.shape[1] - image_xmin
            centered_ymin = ymin
            centered_ymax = ymax
            
        elif image_xmax >= image.shape[1] and image_xmin >= image.shape[1]:
            # detection fully on the right
            image_xmin = image_xmin - image.shape[1]
            image_xmax = image_xmax - image.shape[1]
            centered_image = image[:, image_xmin:image_xmax, :]
            
            centered_xmin = xmin - image.shape[1] - image_xmin
            centered_xmax = xmax - image.shape[1] - image_xmin
            centered_ymin = ymin
            centered_ymax = ymax
        
        else:
            raise ValueError(f"Invalid case : image_xmin = {image_xmin}, image_xmax = {image_xmax}")
        
        centered_box = [centered_xmin, centered_ymin, centered_xmax, centered_ymax]
        
        return centered_image, centered_box, image_xmin, image_xmax

    elif projection_type == "perspective":
        raise NotImplementedError("Perspective projection is not implemented yet")
    else:
        raise ValueError(f"Invalid projection type : {projection_type}")
    
def erode_mask(mask : torch.Tensor, erode_n_pixels : int = 0) -> torch.Tensor:
    """ Perform erosion and dilation on a mask to remove islands and fill holes.

    Args:
        mask (torch.Tensor): Mask to erode and dilate, must be a mask of torch.bool. Shape is (H,W)
        erode_and_dilate_n_pixels (int, optional): Number of pixels to erode and dilate the mask. Defaults to 0.

    Returns:
        torch.Tensor: Eroded and dilated mask, same shape and device as input mask, dtype is torch.bool
    """
    assert(mask.dtype == torch.bool), "Mask must be a boolean tensor got {}".format(mask.dtype)
    mask = mask.float() # convert to float here for conv2d
    
    assert(mask.min() >= 0.0 and mask.max() <= 1.0), "Mask must be a binary mask"
    assert(mask.ndim == 2), "Mask must be a 2D tensor"
    assert(erode_n_pixels % 2 == 1), "Erode n pixels must be odd"
    
    if erode_n_pixels > 0:
        with torch.no_grad():
            mask = mask.unsqueeze(0).unsqueeze(0)
            kernel = torch.ones((erode_n_pixels, erode_n_pixels))
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.to(mask.device)
            mask = torch.nn.functional.conv2d(mask, kernel, padding=erode_n_pixels//2)
            mask = (mask == (erode_n_pixels**2)).squeeze(0).squeeze(0) #.float()
            # mask = torch.nn.functional.conv2d(mask, kernel, padding=erode_n_pixels//2)
            # mask = (mask > 0.0)
            # mask = mask.squeeze(0).squeeze(0)
    
    return mask

def erode_and_dilate_mask(mask : torch.Tensor, erode_and_dilate_n_pixels : int = 0) -> torch.Tensor:
    """ Perform erosion and dilation on a mask to remove islands and fill holes.

    Args:
        mask (torch.Tensor): Mask to erode and dilate, must be a mask of torch.bool. Shape is (H,W)
        erode_and_dilate_n_pixels (int, optional): Number of pixels to erode and dilate the mask. Defaults to 0.

    Returns:
        torch.Tensor: Eroded and dilated mask, same shape and device as input mask, dtype is torch.bool
    """
    assert(mask.dtype == torch.bool), "Mask must be a boolean tensor got {}".format(mask.dtype)
    mask = mask.float() # convert to float here for conv2d
    
    assert(mask.min() >= 0.0 and mask.max() <= 1.0), "Mask must be a binary mask"
    assert(mask.ndim == 2), "Mask must be a 2D tensor"
    assert(erode_and_dilate_n_pixels % 2 == 1), "Erode and dilate n pixels must be odd"
    
    if erode_and_dilate_n_pixels > 0:
        with torch.no_grad():
            mask = mask.unsqueeze(0).unsqueeze(0)
            kernel = torch.ones((erode_and_dilate_n_pixels, erode_and_dilate_n_pixels))
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.to(mask.device)
            mask = torch.nn.functional.conv2d(mask, kernel, padding=erode_and_dilate_n_pixels//2)
            mask = (mask == (erode_and_dilate_n_pixels**2)).float()
            mask = torch.nn.functional.conv2d(mask, kernel, padding=erode_and_dilate_n_pixels//2)
            mask = (mask > 0.0)
            mask = mask.squeeze(0).squeeze(0)
    
    return mask

def get_bounding_box_torch(mask : torch.Tensor) -> tuple:
    """
    Get the bounding box of a mask using torch.
    
    Args:
        mask (torch.Tensor): Mask to get the bounding box of, must be a mask of torch.bool. Shape is (H,W)
                
    Returns:
        tuple: Bounding box coordinates (xmin, ymin, xmax, ymax)
    """
    
    assert(mask.dtype == torch.bool), "Mask must be a boolean tensor got {}".format(mask.dtype)
    mask = mask.float() # convert to float here for conv2d
    
    assert(mask.min() >= 0.0 and mask.max() <= 1.0), "Mask must be a binary mask"
    assert(mask.ndim == 2), "Mask must be a 2D tensor got {}".format(mask.ndim)

    # get the bounding box of the mask
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) == 0 or len(xs) == 0:
        return None, None, None, None

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return xmin.item(), ymin.item(), xmax.item(), ymax.item()

def get_full_image_mask(centered_image_mask : torch.Tensor, centered_image_xmin : int, full_image_size : tuple) -> torch.Tensor:
    """
    Get the full image mask from the centered image mask.
    
    Args:
        centered_image_mask (torch.Tensor): Mask of shape [HCentered, WCentered]
        centered_image_xmin (int): X coordinate of the left edge of the centered image
        full_image_size (tuple): Tuple of (H, W) of the full image size

    Returns:
        torch.Tensor: Mask of shape [H, W] (full image size)
    """
    
    assert(centered_image_mask.dtype == torch.bool), "Centered image mask must be a boolean tensor got {}".format(centered_image_mask.dtype)
    assert(centered_image_mask.ndim == 2), "Centered image mask must be a 2D tensor got {}".format(centered_image_mask.ndim)
    
    centered_image_xmin = centered_image_xmin
    centered_image_xmax = centered_image_xmin + centered_image_mask.shape[1]
    
    full_image_mask = torch.zeros(full_image_size[:2], dtype=torch.bool)

    if centered_image_xmin >= 0 and centered_image_xmax < full_image_size[1]:
        full_image_mask[:, centered_image_xmin:centered_image_xmax] = centered_image_mask
        
    elif centered_image_xmin < 0 and centered_image_xmax >= 0:
        # on the edge (left)
        full_image_mask[:, centered_image_xmin:] = centered_image_mask[:, :-centered_image_xmin]
        full_image_mask[:, :centered_image_xmax] = centered_image_mask[:, -centered_image_xmin:]
        
    elif centered_image_xmax >= full_image_size[1] and centered_image_xmin < full_image_size[1]:
        # on the edge (right)
        full_image_mask[:, centered_image_xmin-full_image_size[1]:] = centered_image_mask[:, :full_image_size[1]-centered_image_xmin]
        full_image_mask[:, :centered_image_xmax-full_image_size[1]] = centered_image_mask[:, full_image_size[1]-centered_image_xmin:]

    else:
        # should not be fully outside
        raise ValueError(f"Invalid case : centered_image_xmin = {centered_image_xmin}, centered_image_xmax = {centered_image_xmax}")
            
    return full_image_mask

def get_full_image_keypoints(vitpose_keypoints_in_centered_image : np.ndarray, centered_image_xmin : int, full_image_size : tuple) -> np.ndarray:
    """
    Get the full image keypoints from the centered image keypoints.
    
    Args:
        vitpose_keypoints_in_centered_image (np.ndarray): Keypoints of shape [NKP, 2]
        centered_image_xmin (int): X coordinate of the left edge of the centered image
        full_image_size (tuple): Tuple of (H, W) of the full image size

    Returns:
        np.ndarray: Keypoints of shape [NKP, 2]
    """
    
    vitpose_keypoints_in_global_image = np.zeros_like(vitpose_keypoints_in_centered_image)
    for i in range(17):
        kp_x = vitpose_keypoints_in_centered_image[i, 0]
        kp_y = vitpose_keypoints_in_centered_image[i, 1]
        
        kp_x_global = kp_x + centered_image_xmin
        if kp_x_global >= full_image_size[1]:
            kp_x_global = kp_x_global - full_image_size[1]
            
        kp_y_global = kp_y

        vitpose_keypoints_in_global_image[i, 0] = kp_x_global
        vitpose_keypoints_in_global_image[i, 1] = kp_y_global
    
    return vitpose_keypoints_in_global_image

def get_centered_image_keypoints(vitpose_keypoints_in_global_image : np.ndarray, centered_image_xmin : int, full_image_size : tuple) -> np.ndarray:
    """
    Get the centered image keypoints from the global image keypoints.
    
    Args:
        vitpose_keypoints_in_global_image (np.ndarray): Keypoints of shape [NKP, 2]
        centered_image_xmin (int): X coordinate of the left edge of the centered image
        full_image_size (tuple): Tuple of (H, W) of the full image size

    Returns:
        np.ndarray: Keypoints of shape [NKP, 2]
    """
    vitpose_keypoints_in_centered_image = np.zeros_like(vitpose_keypoints_in_global_image)
    for i in range(17):
        kp_x = vitpose_keypoints_in_global_image[i, 0]
        kp_y = vitpose_keypoints_in_global_image[i, 1]
        
        kp_x_centered = kp_x - centered_image_xmin
        if kp_x_centered < 0:
            kp_x_centered = kp_x_centered + full_image_size[1]
            
        kp_y_centered = kp_y
        
        vitpose_keypoints_in_centered_image[i, 0] = kp_x_centered
        vitpose_keypoints_in_centered_image[i, 1] = kp_y_centered
    
    return vitpose_keypoints_in_centered_image

    
#### Avoid the very short tracks that are often false detections / duplicates ####
def filter_isolated_detections(detections, window_size = 5, min_element_in_window = 3):
    """ Filter isolated detections (very short appearances).

    Args:
        detections (list): List of detections organized by images : [{"image_idx": ..., "detections": [{"track_id": ..., "xmin": ..., "ymin": ..., "xmax": ..., "ymax": ..., "mask_rle": ...}, ...]}, {...}]
        window_size (int, optional): Window size to check if the track is isolated. Defaults to 5.
        min_element_in_window (int, optional): Minimum number of elements in the window to consider the track not isolated. Defaults to 3.

    Returns:
        _type_: _description_
    """
    track_ids_appearances = {}
    for image_idx, image_det in tqdm(enumerate(detections), total=len(detections)):
        for det in image_det["detections"]:
            track_id = det["track_id"]
            if track_id not in track_ids_appearances:
                track_ids_appearances[track_id] = np.zeros(len(detections), dtype=bool)
            track_ids_appearances[track_id][image_idx] = True
    
    # print(track_ids_appearances.keys())
    
    remove_track_ids_at_image_idxs = []
    for track_id, appearances in track_ids_appearances.items():
        
        min_idx = np.min(np.where(appearances)[0])
        max_idx = np.max(np.where(appearances)[0])
        
        remove_for_track = []
        
        for k in range(min_idx, max_idx+1):
            window = appearances[max(0, k-window_size):min(len(detections), k+window_size)]
            if np.sum(window) < min_element_in_window:
                remove_track_ids_at_image_idxs.append({"track_id": track_id, "image_idx": k})
                remove_for_track.append(k)
                
        prInfo(f"Track {track_id} (min : {min_idx}, max : {max_idx}) -> removed isolated {remove_for_track}")
                    
    for remove_dict in remove_track_ids_at_image_idxs:
        track_to_remove = remove_dict["track_id"]
        index_to_remove = remove_dict["image_idx"]
        
        # don't remove but modify the detections validity
        new_dets_at_idx = []
        for det in detections[index_to_remove]["detections"]:
            if det["track_id"] == track_to_remove:
                det["validity"] = "isolated_{}".format(det["validity"])
            
            new_dets_at_idx.append(det)

        detections[index_to_remove]["detections"] = new_dets_at_idx
        
        # print(f"Removing {track_to_remove} at {index_to_remove}")
        # detections[index_to_remove]["detections"] = [det for det in detections[index_to_remove]["detections"] if det["track_id"] != track_to_remove]

    return detections


UPDATED_INTERACT_ZONE_PROMPT = None

def main(args, video_predictor: SAM2CameraPredictor, image_predictor: SAM2ImagePredictor, vitpose_image_processor: AutoProcessor, vitpose_model: VitPoseForPoseEstimation, episode: str, interact_zone_prompt: list = None):

    global UPDATED_INTERACT_ZONE_PROMPT

    # Load image list
    images_path = os.path.join(args.dir, "episodes", episode, "images_360")
    images_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_path)) for f in fn if (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPG") or f.endswith(".JPEG") or f.endswith(".png") or f.endswith(".PNG"))]
    images_list.sort()
    
    # images_list = images_list[300:550]
    
    assert(len(images_list) > 0), "No images found in the images path : {}".format(images_path)
    prInfo("Found {} images".format(len(images_list)))
    
    # Load detection model from ultralytics
    detection_model = YOLO(f"yolo11{args.yolo_model_size}.pt") #, device = video_predictor.device)  # model for segmantion and detection

    if args.interact_zone_tracking != "":
        
        yolo_detections_output = []
        for image_idx in range(0, len(images_list)):
            current_image_detections = {"image_file": os.path.basename(images_list[image_idx]), "detections": []}
            if image_idx == 0:
                # only add for the first image
                current_image_detections["detections"].append({"xmin": interact_zone_prompt[0],
                                                               "ymin": interact_zone_prompt[1],
                                                               "xmax": interact_zone_prompt[2],
                                                               "ymax": interact_zone_prompt[3]})
            else:
                # empty detections
                pass
            
            yolo_detections_output.append(current_image_detections)

    else:
        # No prompt detections, run YOLO detections
        
        # Initialize output variables for YOLO detections
        yolo_detections_output = []
        image_decomposition_positions = ["left", "front", "right", "back"]

        # ======== YOLO Detections ========
        for image_idx in tqdm(range(0, len(images_list), args.yolo_batch_size), desc="Processing input images (detection with YOLO)"):
            
            # if os.path.exists("filtered_yolo_detections_output.pkl"):
            #     print("DEBUG : SKIPPING YOLO DETECTIONS")
            #     continue
            
            image_files_batch = images_list[image_idx:image_idx+args.yolo_batch_size]

            # Batch output variables
            batch_list = []
            original_images_list = []
            
            # Prepare batch by splitting the image into 4 overlapping subimages
            for image_file in image_files_batch:
                image = Image.open(image_file)
                image = np.array(image) # [H, W, 3] with usually H = 1920, H = 3840 for 360 images
                original_images_list.append(image)
            
                # Create 4 overlapping images
                image_left = image[:, :(image.shape[1]//2), :]
                image_front = image[:, (image.shape[1]//4):(image.shape[1]//4) + (image.shape[1]//2), :]
                image_right = image[:, (image.shape[1]//4)*2:(image.shape[1]//4)*2 + (image.shape[1]//2), :]
                # back image is a composition of the edges
                image_back = np.concatenate([image[:, -(image.shape[1]//4):, :], image[:, :(image.shape[1]//4), :]], axis=1)
                
                batch_list.extend([image_left, image_front, image_right, image_back])
                
                if args.display:
                    
                    image_show = image.copy()
                    image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
                    image_show = cv2.resize(image_show, (image_show.shape[1]//2, image_show.shape[0]//2))
                    cv2.imshow("full image", image_show)
                    key = cv2.waitKey(int(1000/15))
                    if key == ord('q') or key == 27:
                        exit(0)
                    
                    # draw the valid zones of each subimage (the center of the image, if a topleft detection 
                    # falls outside of it we will not use it so it will be processed another camera)
                    image_left = image_left.astype(np.float32) / 255.0
                    image_left[:, :image_left.shape[1]//4, :] *= 0.5
                    image_left[:, image_left.shape[1]//4*3:, :] *= 0.5

                    image_front = image_front.astype(np.float32) / 255.0
                    image_front[:, :image_front.shape[1]//4, :] *= 0.5
                    image_front[:, image_front.shape[1]//4*3:, :] *= 0.5

                    image_right = image_right.astype(np.float32) / 255.0            
                    image_right[:, :image_right.shape[1]//4, :] *= 0.5
                    image_right[:, image_right.shape[1]//4*3:, :] *= 0.5
                    
                    image_back = image_back.astype(np.float32) / 255.0
                    image_back[:, :image_back.shape[1]//4, :] *= 0.5
                    image_back[:, image_back.shape[1]//4*3:, :] *= 0.5
                    
                    stacked = np.concatenate([image_left, image_front, image_right, image_back], axis=0)
                    stacked = cv2.resize(stacked, (stacked.shape[1]//4, stacked.shape[0]//4))
                    stacked = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
                    cv2.imshow("stacked", stacked)
                    key = cv2.waitKey(int(1000/15))
                    if key == ord('q') or key == 27:
                        exit(0)
                    # batch_list.append(stacked)
                
            # Run YOLO detection on the batch
            batch_detections = detection_model(batch_list, verbose=False, device = video_predictor.device) # results of len args.yolo_batch_size * 4
            
            # Rearrange the outputs by original images
            batch_detections = [batch_detections[i:i+4] for i in range(0, len(batch_detections), 4)] # list of list
            batch_list_rearranged = [batch_list[i:i+4] for i in range(0, len(batch_list), 4)] # list of list
            
            # Iterate over original images
            for batch_image_idx, subimages_detections in enumerate(batch_detections):

                original_image = original_images_list[batch_image_idx].copy()
                original_image_width = original_image.shape[1]
                subimage_width = original_image_width // 2
                
                subimages = batch_list_rearranged[batch_image_idx] # list of 4 subimages
                subimages = [subimage.astype(np.float32) / 255.0 for subimage in subimages]

                # YOLO detection element to be appended to the output
                current_image_detections = {"image_file": os.path.basename(image_files_batch[batch_image_idx]), "detections": []}
                
                # Iterate over subimages
                for position, result, subimage in zip(image_decomposition_positions, subimages_detections, subimages):
                    
                    # print(result.names) # dict with {id: NAME, id: NAME, ...}
                    # print(result.boxes.cls) # tensor of class ids
                    # print(result.boxes.conf) # tensor of confidnece scores
                    # print(result.boxes.xyxy) # tensor of shape (N, 4) with coordinates in original image
                
                    if result.boxes is None:
                        continue
                    
                    boxes = result.boxes.xyxy
                    class_ids = result.boxes.cls
                    confidences = result.boxes.conf
                    classes_names =  [result.names[int(class_id.cpu().numpy())] for class_id in class_ids]
                    
                    if args.display:
                        subimage[:, :int(subimage_width//4), :] *= 0.5
                        subimage[:, 3 * int(subimage_width//4):, :] *= 0.5
                        
                    # Iterate over detections
                    for box_idx, (box, class_name, confidence) in enumerate(zip(boxes, classes_names, confidences)):
                        if class_name != "person":
                            continue
                        
                        # if confidence.item() < args.yolo_confidence_threshold:
                        #     continue
                        
                        xmin = box[0].item()
                        ymin = box[1].item()
                        xmax = box[2].item()
                        ymax = box[3].item()
                        
                        if args.display:
                            color = (0, 255, 0)
                            if (xmin < subimage_width//4 or xmin > 3 * (subimage_width//4)):
                                color = (255, 255, 0)
                            if confidence.item() < args.yolo_confidence_threshold:
                                color = (255, 0, 0)
                            cv2.rectangle(subimage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 8)
                            cv2.putText(subimage, class_name, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.circle(subimage, (int(xmin), int(ymin)), 16, color, -1)
                        
                        if xmin < subimage_width//4 or xmin > 3 * (subimage_width//4):
                            # do not consider "edge detections" they will be processed in another camera
                            continue
                        
                        # Adjust the bounding box coordinates to the original image
                        if position == "left":
                            xmin += 0
                            xmax += 0
                        elif position == "front":
                            xmin += original_image_width//4
                            xmax += original_image_width//4
                        elif position == "right":
                            xmin += 2 * original_image_width//4
                            xmax += 2 * original_image_width//4
                        elif position == "back":
                            # weird case because then xmax ends up at the other end of the image but keep it like this for consistency
                            xmin += 3 * original_image_width//4
                            xmax += 3 * original_image_width//4
                        
                        # Append the detection to the output of the current image
                        current_image_detections["detections"].append({"xmin": int(xmin),
                                                                        "ymin": int(ymin),
                                                                        "xmax": int(xmax),
                                                                        "ymax": int(ymax),
                                                                        "class_name": class_name,
                                                                        "confidence": confidence.item(),
                                                                        "validity": "valid" if confidence.item() > args.yolo_confidence_threshold else "low_conf"})
                
                if args.display:
                    stacked = np.concatenate([subimage for subimage in subimages], axis=0)
                    stacked = cv2.resize(stacked, (stacked.shape[1]//4, stacked.shape[0]//4))
                    stacked = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
                    cv2.imshow("stacked with detections", stacked)
                    key = cv2.waitKey(int(1000/15))
                    if key == ord('q') or key == 27:
                        exit(0)
                    # batch_list.append(stacked)
                    
                # Append the detection element to the output of YOLO detections
                yolo_detections_output.append(current_image_detections)

                if args.display:
                    # FOR DEBUGGING
                    # draw box
                    original_image_show = original_image.copy()
                    
                    for detection in current_image_detections["detections"]:
                        color = (0, 255, 0)
                        if detection["confidence"] < args.yolo_confidence_threshold:
                            color = (255, 0, 0)
                        cv2.rectangle(original_image_show, (int(detection["xmin"]), int(detection["ymin"])), (int(detection["xmax"]), int(detection["ymax"])), color, 2)
                        cv2.putText(original_image_show, detection["class_name"], (int(detection["xmin"]), int(detection["ymin"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                    img_show = original_image_show.copy()
                    img_show = cv2.resize(img_show, (1920, 960))
                    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
                    cv2.imshow("full image with detections", img_show)
                    key = cv2.waitKey(0)
                    if key == ord('q') or key == 27:
                        exit(0)
                    # cv2.destroyAllWindows()
    
    if args.interact_zone_tracking != "":
        filtered_yolo_detections_output = yolo_detections_output
    else:
        filtered_yolo_detections_output = []
        for image_idx, image_file in tqdm(enumerate(images_list), desc="Processing input images (image segmentation and first filtering)"):
            
            
            # if os.path.exists("filtered_yolo_detections_output.pkl"):
            #     print("DEBUG : SKIPPING FILTERING")
            #     continue
            
            # skip images for track initialization !
            if image_idx % args.track_initialization_skip_n_images != 0:
                filtered_yolo_detections_output.append({"image_file": image_file, "detections": []})
                continue
            
            yolo_detections_in_image = yolo_detections_output[image_idx]["detections"]
            
            # skip image if no detections
            if len(yolo_detections_in_image) == 0:
                prWarning("No detections in image : {}, skipping".format(image_file))
                filtered_yolo_detections_output.append({"image_file": image_file, "detections": []})
                continue

            # Load the full image        
            image = Image.open(image_file)
            image = np.array(image)
            if args.display:
                centered_images_show = []
                main_image_show = image.copy()
                main_image_show = cv2.cvtColor(main_image_show, cv2.COLOR_RGB2BGR)
                main_image_show = cv2.putText(main_image_show, f"Image {image_idx+1} / {len(images_list)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # create a list of valid detections images and boxes centered for the current image, to split it into batches
            valid_detections_images_centered_for_current_image = []
            valid_detections_boxes_centered_for_current_image = []
            valid_centered_image_xmins_for_current_image = []
            valid_detections_yolo_idxs_for_current_image = []
            for yolo_detection_idx, yolo_detection in enumerate(yolo_detections_in_image):
                
                xmin, xmax, ymin, ymax = yolo_detection["xmin"], yolo_detection["xmax"], yolo_detection["ymin"], yolo_detection["ymax"]
                detection_width_prop_full_image = (xmax - xmin) / image.shape[1]
                detection_height_prop_full_image = (ymax - ymin) / image.shape[0]
                validity = yolo_detection["validity"]
                confidence = yolo_detection["confidence"]
                if args.display:
                    color = COLORS_SEQUENCE_BGR[yolo_detection_idx % len(COLORS_SEQUENCE_BGR)]
                    
                    det_text_1 = f"{yolo_detection_idx} | {validity} | {confidence*100:.0f}%"
                    det_text_2 = f"{detection_width_prop_full_image * 100:.2f}% | {detection_height_prop_full_image * 100:.2f}%"
                    # to draw the box even if on both ends of the image
                    if xmax > main_image_show.shape[1]:
                        if xmin > main_image_show.shape[1]:
                            # fully outside
                            cv2.rectangle(main_image_show, (xmin-main_image_show.shape[1], ymin), (xmax-main_image_show.shape[1], ymax), color, 2)
                            cv2.putText(main_image_show, det_text_1, (xmin-main_image_show.shape[1], ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                            cv2.putText(main_image_show, det_text_2, (xmin-main_image_show.shape[1], ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)
                        else:
                            # two sides
                            cv2.rectangle(main_image_show, (xmin, ymin), (main_image_show.shape[1], ymax), color, 2)
                            cv2.rectangle(main_image_show, (0, ymin), (xmax-main_image_show.shape[1], ymax), color, 2)  
                            cv2.putText(main_image_show, det_text_1, (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                            cv2.putText(main_image_show, det_text_2, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)
                    else:
                        cv2.rectangle(main_image_show, (xmin, ymin), (xmax, ymax), color, 2)                    
                        cv2.putText(main_image_show, det_text_1, (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                        cv2.putText(main_image_show, det_text_2, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

                # skip detection if not valid already
                if validity != "valid":
                    continue
                
                # skip detection if too small
                if detection_width_prop_full_image < (MIN_BOX_SIZE_WIDTH_PROP_OF_CENTERED_IMAGE/2) or detection_height_prop_full_image < (MIN_BOX_SIZE_HEIGHT_PROP_OF_CENTERED_IMAGE/2):
                    # allow half the normal size because we are comparing against a full image !
                    continue
                
                yolo_detection_box = [xmin, ymin, xmax, ymax]
                image_centered, box_in_image_centered, centered_image_xmin, _ = get_image_centered_on_detection(image, [yolo_detection["xmin"], yolo_detection["ymin"], yolo_detection["xmax"], yolo_detection["ymax"]]) # only cropping supported for now

                valid_detections_images_centered_for_current_image.append(image_centered)
                valid_detections_boxes_centered_for_current_image.append(box_in_image_centered)
                valid_centered_image_xmins_for_current_image.append(centered_image_xmin)
                valid_detections_yolo_idxs_for_current_image.append(yolo_detection_idx)

            # skip image if not valid detections
            if len(valid_detections_images_centered_for_current_image) == 0:
                prWarning("No valid detections in image : {}, skipping".format(image_file))
                filtered_yolo_detections_output.append({"image_file": image_file, "detections": []})
                continue
                    
            # Predict the SAM2 image segmentation masks batch by batch
            n_batch_for_image = len(valid_detections_images_centered_for_current_image) // args.sam2_image_batch_size + 1
            valid_detections_processed_for_current_image = {} 
            # Dict of dicts with keys : 
            # {yolo_idx : {
                # "mask_rle_in_global_image": XX, 
                # "mask_area_prop_in_centered_image": XX, 
                # "mask_pixels_in_global_image": XX, 
                # "mask_shape_in_global_image": XX, 
                # "vitpose_keypoints_in_global_image": XX, 
                # "vitpose_scores": XX,
                # "vitpose_kp_in_mask": XX,
                # "vitpose_valid_keypoints": XX,
                # "vitpose_valid_keypoints_in_mask": XX}, ...}
            
            for batch_idx in range(n_batch_for_image):
                batch_images_centered = valid_detections_images_centered_for_current_image[batch_idx*args.sam2_image_batch_size:(batch_idx+1)*args.sam2_image_batch_size]
                batch_boxes_centered = valid_detections_boxes_centered_for_current_image[batch_idx*args.sam2_image_batch_size:(batch_idx+1)*args.sam2_image_batch_size] # [ [xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
                batch_centered_image_xmins = valid_centered_image_xmins_for_current_image[batch_idx*args.sam2_image_batch_size:(batch_idx+1)*args.sam2_image_batch_size]
                batch_yolo_idxs = valid_detections_yolo_idxs_for_current_image[batch_idx*args.sam2_image_batch_size:(batch_idx+1)*args.sam2_image_batch_size]
                batch_vitpose_results = [] # list of dict, [{"keypoints": [NKP, 2], "scores": [NKP]}, ...]
                if len(batch_images_centered) == 0:
                    continue
                
                image_predictor.set_image_batch(batch_images_centered)
                masks_batch, scores_batch, _ = image_predictor.predict_batch(None, None, box_batch=batch_boxes_centered, multimask_output=False, return_torch_tensors=True)
                
                # Pose estimation with ViTPose (in centered images coordinates)
                
                vitpose_boxes = [ [[box[0], box[1], box[2] - box[0], box[3] - box[1]]] for box in batch_boxes_centered] # [ [xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ... ] -> [ [[xmin, ymin, width, height]], [[xmin, ymin, width, height]], ... ]
                vitpose_inputs = vitpose_image_processor(batch_images_centered, boxes=vitpose_boxes, return_tensors="pt").to(torch.device("cuda:0"))
                # print(vitpose_inputs["pixel_values"].shape) # torch.Size([4, 3, 256, 192]) [B, C, H, W] with B = len(batch_images_centered)
            
                with torch.no_grad():
                    vitpose_outputs = vitpose_model(**vitpose_inputs)

                vitpose_results = vitpose_image_processor.post_process_pose_estimation(vitpose_outputs, boxes=vitpose_boxes)  # list of list of dict, [ [{"keypoints": [NKP, 2], "scores": [NKP]}], [{"keypoints": [NKP, 2], "scores": [NKP]}], ...] (list inside is always of length 1, because we only have one box per image)
                for vitpose_result in vitpose_results:
                    batch_vitpose_results.append(vitpose_result[0])
                    
                #del vitpose_inputs, vitpose_outputs
                #torch.cuda.empty_cache()
                
                for centered_input_image, centered_input_box, centered_image_masks, centered_image_xmin, yolo_idx, vitpose_result in zip(batch_images_centered, batch_boxes_centered, masks_batch, batch_centered_image_xmins, batch_yolo_idxs, batch_vitpose_results):
                    centered_image_mask = (centered_image_masks[0] > 0.0) # torch.Tensor of shape H, W, dtype = torch.bool, device = cuda
                    centered_image_mask = erode_and_dilate_mask(centered_image_mask, erode_and_dilate_n_pixels=ERODE_AND_DILATE_N_PIXELS)
                    mask_area_prop_in_centered_image = centered_image_mask.sum() / (centered_input_image.shape[0] * centered_input_image.shape[1])
                    
                    full_image_mask = get_full_image_mask(centered_image_mask, centered_image_xmin, image.shape) # Mask of shape [H, W] (full image size)
                    full_image_mask_rle = encode_RLE(full_image_mask) # 1D torch.Tensor

                    vitpose_keypoints_in_global_image = get_full_image_keypoints(vitpose_result["keypoints"].cpu().numpy(), centered_image_xmin, image.shape)
                    vitpose_valid_keypoints = 0
                    vitpose_valid_keypoints_in_mask = 0
                    vitpose_kp_in_mask = np.zeros_like(vitpose_result["scores"])
                    for kpidx, kp, score in zip(range(len(vitpose_keypoints_in_global_image)), vitpose_keypoints_in_global_image, vitpose_result["scores"]):
                        if score >= 0.5:
                            vitpose_valid_keypoints += 1
                            if full_image_mask[int(kp[1]), int(kp[0])]:
                                vitpose_valid_keypoints_in_mask += 1    
                                vitpose_kp_in_mask[kpidx] = 1

                    if args.display:
                        color = COLORS_SEQUENCE_BGR[yolo_idx % len(COLORS_SEQUENCE_BGR)]
                        input_image_show = centered_input_image.copy()
                        input_image_show = cv2.cvtColor(input_image_show, cv2.COLOR_RGB2BGR)
                        mask_show = centered_image_mask.cpu().numpy()
                        mask_show = (mask_show * 255.0).astype(np.uint8)
                        mask_show = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)
                        input_image_show = cv2.addWeighted(input_image_show, 0.3, mask_show, 0.7, 0)
                        input_image_show = cv2.rectangle(input_image_show, (centered_input_box[0], centered_input_box[1]), (centered_input_box[2], centered_input_box[3]), color, 3)
                        
                        cv2.putText(input_image_show, f"Image {image_idx+1} / {len(images_list)}, batch {batch_idx+1} / {n_batch_for_image}, yolo_idx : {yolo_idx}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
                        validkpcolor = (0, 255, 0) if vitpose_valid_keypoints > MIN_VITPOSE_VALID and vitpose_valid_keypoints_in_mask > MIN_VITPOSE_VALID_IN_MASK else (0, 0, 255)
                        cv2.putText(input_image_show, f"valid_keypoints : {vitpose_valid_keypoints} / valid_keypoints_in_mask : {vitpose_valid_keypoints_in_mask}", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, validkpcolor, 5)
                        validmasksizecolor = (0, 255, 0) if mask_area_prop_in_centered_image > MIN_MASK_PIXELS_PROP_OF_CENTERED_IMAGE else (0, 0, 255)
                        cv2.putText(input_image_show, f"mask_area : {mask_area_prop_in_centered_image * 100:.2f}%", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2.0, validmasksizecolor, 5)
                        for kp, score in zip(vitpose_result["keypoints"], vitpose_result["scores"]):
                            if score < 0.5:
                                cv2.circle(input_image_show, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), 2)
                            else:
                                cv2.circle(input_image_show, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                        
                        centered_images_show.append(input_image_show)

                        full_image_mask_show = main_image_show.copy()
                        full_image_mask_show[full_image_mask.cpu().numpy()] = 255
                        full_image_mask_show = full_image_mask_show.astype(np.uint8)
                        
                        for kp, score in zip(vitpose_keypoints_in_global_image, vitpose_result["scores"]):
                            if score < 0.5:
                                cv2.circle(full_image_mask_show, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), 2)
                            else:
                                cv2.circle(full_image_mask_show, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                        
                        main_image_show = cv2.addWeighted(main_image_show, 0.5, full_image_mask_show, 0.5, 0)
                        
                        
                    valid_detections_processed_for_current_image[yolo_idx] = {"mask_rle_in_global_image": full_image_mask_rle,
                                                                                "mask_area_prop_in_centered_image": mask_area_prop_in_centered_image,
                                                                                "mask_pixels_in_global_image": full_image_mask.sum(),
                                                                                "mask_shape_in_global_image": full_image_mask.shape,
                                                                                "vitpose_keypoints_in_global_image": vitpose_keypoints_in_global_image,
                                                                                "vitpose_scores": vitpose_result["scores"],
                                                                                "vitpose_kp_in_mask": vitpose_kp_in_mask,
                                                                                "vitpose_valid_keypoints": vitpose_valid_keypoints,
                                                                                "vitpose_valid_keypoints_in_mask": vitpose_valid_keypoints_in_mask}
            
            filtered_yolo_detections_output_current_detections = []
            for yolo_idx, detection_processed in valid_detections_processed_for_current_image.items():
                original_yolo_detection = yolo_detections_in_image[yolo_idx]
                
                mask_rle_in_global_image = detection_processed["mask_rle_in_global_image"]
                mask_area_prop_in_centered_image = detection_processed["mask_area_prop_in_centered_image"]
                mask_pixels_in_global_image = detection_processed["mask_pixels_in_global_image"]
                mask_shape_in_global_image = detection_processed["mask_shape_in_global_image"]
                vitpose_keypoints_in_global_image = detection_processed["vitpose_keypoints_in_global_image"]
                vitpose_scores = detection_processed["vitpose_scores"]
                vitpose_kp_in_mask = detection_processed["vitpose_kp_in_mask"]
                vitpose_valid_keypoints = detection_processed["vitpose_valid_keypoints"]
                vitpose_valid_keypoints_in_mask = detection_processed["vitpose_valid_keypoints_in_mask"]
                
                # Add some filtering here !
                if mask_area_prop_in_centered_image > MIN_MASK_PIXELS_PROP_OF_CENTERED_IMAGE and vitpose_valid_keypoints > MIN_VITPOSE_VALID and vitpose_valid_keypoints_in_mask > MIN_VITPOSE_VALID_IN_MASK:
                    
                    original_yolo_detection["mask_rle"] = mask_rle_in_global_image
                    original_yolo_detection["mask_area_prop_in_centered_image"] = mask_area_prop_in_centered_image
                    original_yolo_detection["mask_pixels_in_global_image"] = mask_pixels_in_global_image
                    original_yolo_detection["mask_shape"] = mask_shape_in_global_image
                    original_yolo_detection["vitpose_keypoints_in_global_image"] = vitpose_keypoints_in_global_image
                    original_yolo_detection["vitpose_scores"] = vitpose_scores
                    original_yolo_detection["vitpose_kp_in_mask"] = vitpose_kp_in_mask
                    original_yolo_detection["vitpose_valid_keypoints"] = vitpose_valid_keypoints
                    original_yolo_detection["vitpose_valid_keypoints_in_mask"] = vitpose_valid_keypoints_in_mask
                    
                    filtered_yolo_detections_output_current_detections.append(original_yolo_detection)
            
            filtered_yolo_detections_output.append({"image_file": image_file, "detections": filtered_yolo_detections_output_current_detections})
            
            if args.display:
                
                # show main image
                main_image_show = cv2.resize(main_image_show, (1920, 960)) # resize for display
                cv2.imshow("SAM2 Image Initial Segmentation", main_image_show)
                key = cv2.waitKey(100)
                if key == ord('q') or key == 27:
                    exit(0)
                    
                # show a mosaic of up to 16 images centered
                centered_images_show = [cv2.resize(centered_image_show, (480, 480)) for centered_image_show in centered_images_show]
                mosaic_image_show = np.zeros((1920, 1920, 3), dtype=np.uint8)
                for i in range(min(len(centered_images_show), 16)):
                    row = i // 4
                    col = i % 4
                    mosaic_image_show[row*480:(row+1)*480, col*480:(col+1)*480, :] = centered_images_show[i]
                    #add thin separations
                    mosaic_image_show[row*480:row*480+10, col*480:(col+1)*480, :] = 0
                    mosaic_image_show[(row+1)*480-10:(row+1)*480, col*480:(col+1)*480, :] = 0
                    mosaic_image_show[row*480:(row+1)*480, col*480:col*480+10, :] = 0
                    mosaic_image_show[row*480:(row+1)*480, (col+1)*480-10:(col+1)*480, :] = 0

                cv2.imshow("Mosaic of centered images", mosaic_image_show)
                key = cv2.waitKey(100)
                if key == ord('q') or key == 27:
                    exit(0)
        
        # if os.path.exists("filtered_yolo_detections_output.pkl"):
        #     filtered_yolo_detections_output = pickle.load(open("filtered_yolo_detections_output.pkl", "rb"))
        #     print("LOADED FILTERED YOLO DETECTIONS OUTPUT FROM PICKLE")
        # else:
        #     pickle.dump(filtered_yolo_detections_output, open("filtered_yolo_detections_output.pkl", "wb"))
        
    assert(len(filtered_yolo_detections_output) == len(images_list)), "Number of filtered detections and images should be the same got {} and {}".format(len(filtered_yolo_detections_output), len(images_list))
    
    # ======== SAM2 Segmentation and Tracking ========
    video_predictor_initialized = False
    sam2_detections_output = {} # dict in the form {idx: {"image_file": FILENAME, "detections": [...]}, "idx": ...}, transformed to list before exporting
    current_sam2_track_id = 0
    # Run SAM2 segmentation and tracking
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        
        # Iterate over all images...
        for image_idx, image_file in tqdm(enumerate(images_list), desc="Processing input images (tracking with SAM2)"):            
            yolo_detections_in_image = filtered_yolo_detections_output[image_idx]["detections"]
            if len(yolo_detections_in_image) == 0:
                # before skipping, create the entry in the sam2_detections_output if it does not exist
                if image_idx not in sam2_detections_output:
                    sam2_detections_output[image_idx] = {"image_file": image_file, "detections": []}
                # prWarning("No detections in image : {}, skipping".format(image_file))
                continue

            # Load the full image
            image = Image.open(image_file)
            image = np.array(image) # [H, W, 3] with usually H = 1920, H = 3840 for 360 images
                        
            # ... and over each YOLO detection in the image ...
            for yolo_detection_idx, yolo_detection in enumerate(yolo_detections_in_image):
                                
                # Create the entry in the sam2_detections_output if it does not exist
                if image_idx not in sam2_detections_output:
                    sam2_detections_output[image_idx] = {"image_file": image_file, "detections": []}
                                
                # Check if the initial mask matches with any existing SAM2 track
                yolo_detection_box = [yolo_detection["xmin"], yolo_detection["ymin"], yolo_detection["xmax"], yolo_detection["ymax"]]
                
                if args.interact_zone_tracking == "":
                    # Normal case
                    current_mask = decode_RLE(yolo_detection["mask_rle"], shape = image.shape[:2])
                    matching_sam2_track = None
                    matching_sam2_track_detection_idx = None
                    for detection_comp_idx, detection_comp in enumerate(sam2_detections_output[image_idx]["detections"]):
                        detection_comp_box = [detection_comp["xmin"], detection_comp["ymin"], detection_comp["xmax"], detection_comp["ymax"]]
                        iou_boxes = bb_intersection_over_union_with_shift(yolo_detection_box, detection_comp_box, shift_test=image.shape[1])
                        if iou_boxes > MIN_OVERLAP_IOU_BOXES:
                            detection_comp_mask_rle = detection_comp["mask_rle"]
                            detection_comp_mask = decode_RLE(detection_comp_mask_rle, shape = image.shape[:2])
                            iou_masks = get_iou_torch(detection_comp_mask, current_mask)
                            prop_of_mask_in_mask_comp = prop_of_mask1_in_mask2_torch(current_mask, detection_comp_mask) # check if current_mask is fully enclosed in detection_comp_mask
                            if iou_masks > MIN_OVERLAP_IOU_MASKS or prop_of_mask_in_mask_comp > MIN_MASK_IN_MASK_COMP_PROP:
                                matching_sam2_track = detection_comp["track_id"]
                                matching_sam2_track_detection_idx = detection_comp_idx
                                break
                    
                    if matching_sam2_track is not None:
                        # prWarning(f"[{image_idx}][YOLO detection {yolo_detection_idx}] Matching SAM2 track {matching_sam2_track} (index {matching_sam2_track_detection_idx}) found. Do not pursue tracking from this detection (iou_boxes = {iou_boxes * 100:.2f}%, iou_masks = {iou_masks * 100:.2f}%, prop_of_mask_in_mask_comp = {prop_of_mask_in_mask_comp * 100:.2f}%)")
                        continue
                    else:
                        prSuccess(f"[{image_idx}][YOLO detection {yolo_detection_idx}] No matching SAM2 track found. Pursuing tracking from this detection starting at image {image_idx}. Assigned SAM2 track ID {current_sam2_track_id}")
                        
                    #### Start tracking from the initial frame ####
                    image_centered, box_in_image_centered, centered_image_xmin, _ = get_image_centered_on_detection(image, yolo_detection_box) # only cropping supported for now
                    vitpose_keypoints_in_centered_image = get_centered_image_keypoints(yolo_detection["vitpose_keypoints_in_global_image"], centered_image_xmin, image_centered.shape)
                    if args.use_keypoints_prompt:
                        # NOTE : it does not work so good especially since some keypoints often fall on the edge of the body
                        points_prompt = []
                        for kp, kp_in_mask, score in zip(vitpose_keypoints_in_centered_image, yolo_detection["vitpose_kp_in_mask"], yolo_detection["vitpose_scores"]):
                            if kp_in_mask and score > 0.5:
                                points_prompt.append([kp[0], kp[1]])
                        points_prompt = np.array(points_prompt)
                        points_prompt = torch.from_numpy(points_prompt)
                        labels_prompt = torch.ones(points_prompt.shape[0], dtype=torch.int32) # all the same instance
                        if points_prompt.shape[0] == 0:
                            points_prompt = None
                            labels_prompt = None
                    else:
                        points_prompt = None
                        labels_prompt = None
                else:
                    # when interact zone tracking is used, we do not need all these checks
                    image_centered, box_in_image_centered, centered_image_xmin, _ = get_image_centered_on_detection(image, yolo_detection_box, desired_width=2880) # only cropping supported for now
                    points_prompt = None
                    labels_prompt = None
                    
                # If the video predictor is initialized, reset it before starting a new track
                if video_predictor_initialized:
                    prInfo("Resetting video predictor")
                    video_predictor.reset_state()
                
                # Start like a new tracking with SAM2
                video_predictor.load_first_frame(image_centered)
                video_predictor_initialized = True
                
                # Get the initial mask from SAM2
                # always pretend we are at ID 0 and frame 0 since we are restarting from scratch the tracker
                # use box_in_image_centered as prompt (TODO : could be improved, with pose keypoints for example, and optimized by doing it first for all YOLO detections)
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_prompt(frame_idx=0, obj_id=0, bbox=box_in_image_centered, points=points_prompt, labels=labels_prompt)
                
                centered_image_mask = (out_mask_logits[0, 0, :, :] > 0.0) # Mask of shape [HCentered, WCentered]
                centered_image_mask = erode_and_dilate_mask(centered_image_mask, erode_and_dilate_n_pixels=ERODE_AND_DILATE_N_PIXELS)
                if args.interact_zone_tracking != "":
                    # get the convex hull
                    centered_image_mask = get_convex_hull_mask(centered_image_mask)
                    centered_image_mask = erode_mask(centered_image_mask, erode_n_pixels=15)
                
                mask_area_prop = centered_image_mask.sum() / (image_centered.shape[0] * image_centered.shape[1])
                if mask_area_prop < MIN_MASK_PIXELS_PROP_OF_CENTERED_IMAGE:
                    prWarning(f"[{image_idx}][YOLO detection {yolo_detection_idx}] Mask area is too small ({mask_area_prop * 100:.2f}%). Skipping.")
                    continue
                
                bbox_in_centered_image = get_bounding_box_torch(centered_image_mask)
                bbox_in_absolute_image = [bbox_in_centered_image[0] + centered_image_xmin, 
                                          bbox_in_centered_image[1], 
                                          bbox_in_centered_image[2] + centered_image_xmin, 
                                          bbox_in_centered_image[3]] # this box can have negative coordinates or above the image shape

                full_image_mask = get_full_image_mask(centered_image_mask, centered_image_xmin, image.shape) # Mask of shape [H, W] (full image size)
                full_image_mask_rle = encode_RLE(full_image_mask)
                
                # save the detection for the current sam2 track
                sam2_detections_output[image_idx]["detections"].append({"track_id": current_sam2_track_id,
                                                                        "xmin": bbox_in_absolute_image[0],
                                                                        "ymin": bbox_in_absolute_image[1],
                                                                        "xmax": bbox_in_absolute_image[2],
                                                                        "ymax": bbox_in_absolute_image[3],
                                                                        "validity": "valid",
                                                                        "mask_size": full_image_mask.sum().item(),
                                                                        "mask_rle": full_image_mask_rle})
                
                # ... if it did not match, iterate on all the following images and keep checking if it does not match with any existing detection, otherwise save as new elements of the track ...
                break_count = 0
                for following_image_idx in tqdm(range(image_idx+1, len(images_list)), desc="[{}] Tracking with SAM2".format(current_sam2_track_id)):
                    following_image_file = images_list[following_image_idx]
                    following_image = Image.open(following_image_file)
                    following_image = np.array(following_image)
                    
                    # Center based on the bounding box of the previous detection
                    desired_width = 2880 if args.interact_zone_tracking != "" else None
                    following_image_centered, following_box_in_image_centered, following_image_xmin, following_image_xmax = get_image_centered_on_detection(following_image, bbox_in_absolute_image, desired_width=desired_width)
                                        
                    # Find the mask in the following image
                    tic = time.time()
                    out_obj_ids, out_mask_logits = video_predictor.track(following_image_centered)
                    tac = time.time()
                    if args.verbose or args.display:
                        prTimer(f"Track with SAM2", tic, tac)
                    centered_image_mask = (out_mask_logits[0, 0, :, :] > 0.0)
                    centered_image_mask = erode_and_dilate_mask(centered_image_mask, erode_and_dilate_n_pixels=ERODE_AND_DILATE_N_PIXELS)
                    if args.interact_zone_tracking != "":
                        # get the convex hull
                        centered_image_mask = get_convex_hull_mask(centered_image_mask)
                        centered_image_mask = erode_mask(centered_image_mask, erode_n_pixels=15)
                        
                    bbox_in_centered_image = get_bounding_box_torch(centered_image_mask)
                    
                    # If the mask is empty, skip or break after a while
                    bbox_width_prop = (bbox_in_centered_image[2] - bbox_in_centered_image[0]) / following_image_centered.shape[1] if bbox_in_centered_image[0] is not None else 0
                    bbox_height_prop = (bbox_in_centered_image[3] - bbox_in_centered_image[1]) / following_image_centered.shape[0] if bbox_in_centered_image[1] is not None else 0
                    mask_area_prop = centered_image_mask.sum() / (following_image_centered.shape[0] * following_image_centered.shape[1])
                    if (bbox_in_centered_image[0] is None 
                        or mask_area_prop < MIN_MASK_PIXELS_PROP_OF_CENTERED_IMAGE 
                        or bbox_width_prop < MIN_BOX_SIZE_WIDTH_PROP_OF_CENTERED_IMAGE 
                        or bbox_height_prop < MIN_BOX_SIZE_HEIGHT_PROP_OF_CENTERED_IMAGE):
                        
                        break_count += 1
                        if break_count > MAX_OCCLUSION_FRAMES_COUNT:
                            break_count = 0
                            prWarning(f"Empty mask or too small (bbox_width_prop = {bbox_width_prop * 100:.2f}%, bbox_height_prop = {bbox_height_prop * 100:.2f}%, prop_of_mask = {mask_area_prop * 100:.2f}%) for SAM2 track id {current_sam2_track_id} at image {following_image_idx}, stop the track")
                            break
                        else:
                            if bbox_in_centered_image[0] is None:
                                # prWarning(f"Empty mask (bbox_width_prop = {bbox_width_prop * 100:.2f}%, bbox_height_prop = {bbox_height_prop * 100:.2f}%, prop_of_mask = {mask_area_prop * 100:.2f}%) for SAM2 track id {current_sam2_track_id} at image {following_image_idx}, skipping and continuing the track")
                                continue
                            else:
                                # still do the tracking but increment the counter, so after a while being to small it will break
                                # prWarning(f"Too small (bbox_width_prop = {bbox_width_prop * 100:.2f}%, bbox_height_prop = {bbox_height_prop * 100:.2f}%, prop_of_mask = {mask_area_prop * 100:.2f}%) for SAM2 track id {current_sam2_track_id} at image {following_image_idx}, keep tracking but increment the counter")
                                pass
                    else:
                        break_count = 0
                        
                    bbox_in_absolute_image = [bbox_in_centered_image[0] + following_image_xmin, 
                                              bbox_in_centered_image[1], 
                                              bbox_in_centered_image[2] + following_image_xmin, 
                                              bbox_in_centered_image[3]] # this box can have negative coordinates or above the image shape
                    full_image_mask = get_full_image_mask(centered_image_mask, following_image_xmin, following_image.shape)
                    full_image_mask_rle = encode_RLE(full_image_mask)
                    
                    # Create the entry in the sam2_detections_output if it does not exist
                    if following_image_idx not in sam2_detections_output:
                        sam2_detections_output[following_image_idx] = {"image_file": following_image_file, "detections": []}
                    
                    # Check if the detection matches with any existing SAM2 track
                    matching_sam2_track = None
                    matching_sam2_track_detection_idx = None
                    for detection_comp_idx, detection_comp in enumerate(sam2_detections_output[following_image_idx]["detections"]):
                        detection_comp_box = [detection_comp["xmin"], detection_comp["ymin"], detection_comp["xmax"], detection_comp["ymax"]]
                        iou_boxes = bb_intersection_over_union_with_shift(bbox_in_absolute_image, detection_comp_box, shift_test=following_image.shape[1])
                        if iou_boxes > MIN_OVERLAP_IOU_BOXES:
                            detection_comp_mask_rle = detection_comp["mask_rle"]
                            detection_comp_mask = decode_RLE(detection_comp_mask_rle, shape = following_image.shape[:2])
                            iou_masks = get_iou_torch(detection_comp_mask, full_image_mask)
                            prop_of_mask_in_mask_comp = prop_of_mask1_in_mask2_torch(full_image_mask, detection_comp_mask) # if we are fully enclosed in the compared mask, we break
                            if iou_masks > MIN_OVERLAP_IOU_MASKS or prop_of_mask_in_mask_comp > MIN_MASK_IN_MASK_COMP_PROP:
                                matching_sam2_track = detection_comp["track_id"]
                                matching_sam2_track_detection_idx = detection_comp_idx
                                break
                    
                    # TODO : compare with YOLO and reprompt with keypoints (if it seems to be a good idea...)
                    
                    # If it matches with an existing SAM2 track, break
                    if matching_sam2_track is not None:
                        prWarning(f"Matching SAM2 track {matching_sam2_track} (idx {matching_sam2_track_detection_idx}) found while tracking with SAM2 track id {current_sam2_track_id} at image {following_image_idx}, do not pursue tracking from this detection (iou_boxes = {iou_boxes * 100:.2f}%, iou_masks = {iou_masks * 100:.2f}%, prop_of_mask_in_mask_comp = {prop_of_mask_in_mask_comp * 100:.2f}%)")
                        break
         
                    # Save the detection for the current sam2 track
                    sam2_detections_output[following_image_idx]["detections"].append({"track_id": current_sam2_track_id,
                                                                                        "xmin": bbox_in_absolute_image[0],
                                                                                        "ymin": bbox_in_absolute_image[1],
                                                                                        "xmax": bbox_in_absolute_image[2],
                                                                                        "ymax": bbox_in_absolute_image[3],
                                                                                        "validity": "valid",
                                                                                        "mask_size": full_image_mask.sum().item(),
                                                                                        "mask_rle": full_image_mask_rle})

                    if args.display:
                        # draw centered image
                        image_centered_show = following_image_centered.copy()
                        mask_image = image_centered_show.copy()
                        mask_image[centered_image_mask.cpu().numpy()] = COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)]
                        image_centered_show = cv2.addWeighted(image_centered_show, 0.5, mask_image, 0.5, 0)
                        cv2.putText(image_centered_show, f"ID {current_sam2_track_id} (image {following_image_idx})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                        cv2.putText(image_centered_show, f"bbox_width_prop = {bbox_width_prop * 100:.2f}%, bbox_height_prop = {bbox_height_prop * 100:.2f}%, prop_of_mask = {mask_area_prop * 100:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                        cv2.rectangle(image_centered_show, (int(bbox_in_centered_image[0]), int(bbox_in_centered_image[1])), (int(bbox_in_centered_image[2]), int(bbox_in_centered_image[3])), COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                        image_centered_show = cv2.cvtColor(image_centered_show, cv2.COLOR_BGR2RGB)
                        image_centered_show = cv2.resize(image_centered_show, (480, 480))
                        hostname = socket.gethostname()
                        if "raphael" in hostname:
                            cv2.imshow("image_centered_show", image_centered_show)
                            key = cv2.waitKey(10)
                            if key == ord('q') or key == 27:
                                exit(0)
                        
                        # draw full image
                        image_full_show = following_image.copy()
                        mask_image = image_full_show.copy()
                        mask_image[full_image_mask.cpu().numpy()] = COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)]
                        image_full_show = cv2.addWeighted(image_full_show, 0.5, mask_image, 0.5, 0)
                        detections_in_image = sam2_detections_output[following_image_idx]["detections"]
                        for detection in detections_in_image:
                            color_detection = COLORS_SEQUENCE_BGR[detection["track_id"] % len(COLORS_SEQUENCE_BGR)]
                            associated_mask = decode_RLE(detection["mask_rle"], shape = following_image.shape[:2])
                            associated_mask_iou = get_iou_torch(associated_mask, full_image_mask)
                            associated_mask_image = image_full_show.copy()
                            associated_mask_image[associated_mask.cpu().numpy()] = color_detection
                            image_full_show = cv2.addWeighted(image_full_show, 0.5, associated_mask_image, 0.5, 0)
                            cv2.rectangle(image_full_show, (int(detection["xmin"]), int(detection["ymin"])), (int(detection["xmax"]), int(detection["ymax"])), color_detection, 2)
                            cv2.putText(image_full_show, "ID {} (iou = {:.2f}%)".format(detection["track_id"], associated_mask_iou * 100), (int(detection["xmin"]), int(detection["ymin"]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2)
                        cv2.putText(image_full_show, f"ID {current_sam2_track_id} (image {following_image_idx})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                        cv2.putText(image_full_show, f"bbox_width_prop = {bbox_width_prop * 100:.2f}%, bbox_height_prop = {bbox_height_prop * 100:.2f}%, prop_of_mask = {mask_area_prop * 100:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                        cv2.rectangle(image_full_show, (int(bbox_in_absolute_image[0]), int(bbox_in_absolute_image[1])), (int(bbox_in_absolute_image[2]), int(bbox_in_absolute_image[3])), COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                        image_full_show = cv2.cvtColor(image_full_show, cv2.COLOR_BGR2RGB)
                                            
                        track_save_debug_dir = os.path.join(args.dir, "episodes", episode, "track_debug", "track_{:06d}".format(current_sam2_track_id))
                        if not os.path.isdir(track_save_debug_dir):
                            os.makedirs(track_save_debug_dir)
                        cv2.imwrite(os.path.join(track_save_debug_dir, f"image_full_{following_image_idx:06d}.jpg"), image_full_show)
                        
                        hostname = socket.gethostname()
                        if "raphael" in hostname:
                            image_full_show = cv2.resize(image_full_show, (960, 480))
                            cv2.imshow("image_full_show", image_full_show)
                            key = cv2.waitKey(10)
                            if key == ord('q') or key == 27:
                                exit(0)
                
                # Increment the current SAM2 track id before next tracking
                current_sam2_track_id += 1
    
    # reorder based on keys
    sam2_detections_output = dict(sorted(sam2_detections_output.items()))
        
    # Transform the dict to a list
    sam2_detections_output_list = []
    for image_idx, image_detections in sam2_detections_output.items():
        for det in image_detections["detections"]:
            mask_rle_list = det["mask_rle"].cpu().numpy().tolist()
            det["mask_rle"] = ",".join([str(int(x)) for x in mask_rle_list])
        sam2_detections_output_list.append(image_detections)
    
    # Filter isolated detections
    sam2_detections_output_list = filter_isolated_detections(sam2_detections_output_list)

    assert(len(sam2_detections_output_list) == len(images_list)), "Number of detections and images should be the same"
        
    # Write the detections to a JSON file
    if args.interact_zone_tracking != "":
        write_dic_to_json_file(sam2_detections_output_list, os.path.join(args.dir, "episodes", episode, "interact_zone_tracking_360.json"))
        last_detection_element = sam2_detections_output_list[-1]["detections"]
        assert(len(last_detection_element) == 1), f"There should be one and only one detection element (got {len(last_detection_element)})"
        new_interact_zone_prompt = [last_detection_element[0]["xmin"], last_detection_element[0]["ymin"], last_detection_element[0]["xmax"], last_detection_element[0]["ymax"]]
        prInfo(f"Updating interact zone prompt for the next episode from {interact_zone_prompt} to {new_interact_zone_prompt}")
        UPDATED_INTERACT_ZONE_PROMPT = new_interact_zone_prompt
    else:
        write_dic_to_json_file(sam2_detections_output_list, os.path.join(args.dir, "episodes", episode, "sam2_detections_360.json"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./demo/output/2025_06_13_11_15_31_Theta360", help="Directory to use (default: %(default)s)")
    parser.add_argument("--episodes", "-e", type=str, default="all", help="Episodes to use, separated by commas or 'all' (default: %(default)s)")
    parser.add_argument("--sam2_model", "-sm", type=str, default="tiny", help="SAM2 model to use (tiny, small, base, large) (default: %(default)s)")
    parser.add_argument("--yolo_model_size", "-ys", type=str, default="x", help="YOLO model size to use (n, s, l, x) (default: %(default)s)")
    parser.add_argument("--yolo_confidence_threshold", "-yct", type=float, default=0.7, help="YOLO confidence threshold (default: %(default)s)")
    parser.add_argument("--yolo_batch_size", "-ybs", type=int, default=16, help="Batch size for the YOLO detection (default: %(default)s)")
    parser.add_argument("--sam2_image_batch_size", "-sibs", type=int, default=8, help="Batch size for the SAM2 image prediction (default: %(default)s)")
    parser.add_argument("--track_initialization_skip_n_images", "-tisni", type=int, default=5, help="Skip the first n images for track initialization (default: %(default)s)")
    parser.add_argument("--use_keypoints_prompt", "-ukp", action="store_true", default=False, help="Use the keypoints prompt (default: %(default)s)")
    parser.add_argument("--vos_optimized", "-vos", action="store_true", default=False, help="Use the VOS optimized model (default: %(default)s)")
    parser.add_argument("--display", "-disp", action="store_true", default=False, help="Display the detections live (default: %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose mode (default: %(default)s)")
    parser.add_argument("--interact_zone_tracking", "-izt", type=str, default="", help="Don't track the people but the zone to interact with, give a prompt at the beggining ([xmin, ymin, xmax, ymax] in pixels) and track from there (default: %(default)s)")
    # parser.add_argument("--ngpu", "-ngpu", type=int, default=0, help="Number of GPUs to use, default 0 means use all GPUs")
    args = parser.parse_args()

    # assert(args.erode_masks % 2 == 1), "Erode masks must be odd"
    
    prInfo("Using SAM2 model : {}".format(args.sam2_model))
    prInfo("Using YOLO model size : {}".format(args.yolo_model_size))
    
    sam2_checkpoint = SAM2_MODEL_TO_CHECKPOINT[args.sam2_model]
    sam2_cfg = SAM2_MODEL_TO_CFG[args.sam2_model]
    
    assert os.path.isfile(sam2_checkpoint), "Checkpoint should exist : {}".format(sam2_checkpoint)
    # assert os.path.isfile(os.path.join(here, "sam2",sam2_cfg)), "SAM2 config should exist : {}".format(sam2_cfg)

    # List of episodes to process
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()

    for episode in episodes_list:
        images_path = os.path.join(args.dir, "episodes", episode, "images_360")
        assert os.path.isdir(images_path), "Directory should exist : {}".format(images_path)

    video_predictor = build_sam2_camera_predictor(sam2_cfg, sam2_checkpoint, device=torch.device("cuda", index=0), vos_optimized=args.vos_optimized)
    prSuccess("SAM2 model built successfully!")
    
    image_predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, sam2_checkpoint))
    prSuccess("SAM2 Image Predictor built successfully!")
    
    vitpose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", cache_dir=os.path.join(here, "..", "checkpoints"))
    vitpose_model.to(torch.device("cuda:0"))
    prSuccess("ViTPose Image processor and model loaded")
    
    if args.interact_zone_tracking != "":
        # Use a prompt for the first bbox provided in the args
        interact_zone_prompt = args.interact_zone_tracking.split(",")
        assert len(interact_zone_prompt) == 4, "Interact zone prompt should be 4 values separated by commas"
        interact_zone_prompt = [int(x) for x in interact_zone_prompt]
        prInfo(f"Using interact zone prompt : {interact_zone_prompt} for the first image")
    else:
        interact_zone_prompt = None
    
    for episode in episodes_list:
        main(args, video_predictor, image_predictor, vitpose_image_processor, vitpose_model, episode, interact_zone_prompt)

        if args.interact_zone_tracking != "":
            prInfo(f"Updated interact zone prompt for the next episode from {interact_zone_prompt} to {UPDATED_INTERACT_ZONE_PROMPT}")
            interact_zone_prompt = UPDATED_INTERACT_ZONE_PROMPT