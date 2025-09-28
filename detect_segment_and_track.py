import torch
from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_camera_predictor import SAM2CameraPredictor
import os
import sys
import subprocess
import json
import shutil
import time
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import functional as F
from ultralytics import YOLO

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *

import argparse
from misc.rle_tools import encode_RLE, decode_RLE

SAM2_MODEL_TO_CHECKPOINT = {
    "tiny": os.path.join(here, "checkpoints/sam2.1_hiera_tiny.pt"),
    "tiny_512": os.path.join(here, "checkpoints/sam2.1_hiera_tiny.pt"),
    "small": os.path.join(here, "checkpoints/sam2.1_hiera_small.pt"),
    "base": os.path.join(here, "checkpoints/sam2.1_hiera_base_plus.pt"),
    "large": os.path.join(here, "checkpoints/sam2.1_hiera_large.pt")
}

SAM2_MODEL_TO_CFG = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "tiny_512": "configs/sam2.1/sam2.1_hiera_t_512.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml"
}

MAX_OCCLUSION_FRAMES_COUNT = 60
MIN_OVERLAP_IOU_BOXES = 0.5
MIN_OVERLAP_IOU_MASKS = 0.8
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

def get_image_centered_on_detection(image, detection_box, projection_type = "horizontal_crop"):
    
    # TODO : improve this function with the possibility of doing a proper perspective projection
    
    if projection_type == "horizontal_crop":
        
        xmin = detection_box[0]
        ymin = detection_box[1]
        xmax = detection_box[2]
        ymax = detection_box[3]
        center_box_x = (xmin + xmax) / 2
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
    


def erode_and_dilate_mask(mask, erode_and_dilate_n_pixels = 0):
    """_summary_

    Args:
        mask (torch.Tensor): Mask to erode and dilate, must be a mask of torch.bool. Shape is (H,W)
        erode_and_dilate_n_pixels (int, optional): Number of pixels to erode and dilate the mask. Defaults to 0.

    Returns:
        torch.Tensor: Eroded and dilated mask, same shape and device as input mask, dtype is torch.bool
    """
    assert(mask.dtype == torch.bool), "Mask must be a boolean tensor"
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

def get_bounding_box_torch(mask):
    """
    Get the bounding box of a mask using torch.
    
    Args:
        mask (torch.Tensor): Mask to get the bounding box of, must be a mask of torch.bool. Shape is (H,W)
                
    Returns:
        tuple: Bounding box coordinates (xmin, ymin, xmax, ymax)
    """
    
    assert(mask.dtype == torch.bool), "Mask must be a boolean tensor"
    mask = mask.float() # convert to float here for conv2d
    
    assert(mask.min() >= 0.0 and mask.max() <= 1.0), "Mask must be a binary mask"
    assert(mask.ndim == 2), "Mask must be a 2D tensor"

    # get the bounding box of the mask
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) == 0 or len(xs) == 0:
        return None, None, None, None

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return xmin.item(), ymin.item(), xmax.item(), ymax.item()

def get_full_image_mask(centered_image_mask, centered_image_xmin, full_image_size):
    """
    Get the full image mask from the centered image mask.
    """
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

def main(args, video_predictor: SAM2CameraPredictor):

    images_path = os.path.join(args.dir, "images_360")
    images_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_path)) for f in fn if (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPG") or f.endswith(".JPEG") or f.endswith(".png") or f.endswith(".PNG"))]
    images_list.sort()
    
    # images_list = images_list[300:350]
    
    assert(len(images_list) > 0), "No images found in the images path : {}".format(images_path)
    prInfo("Found {} images".format(len(images_list)))
    
    # Detection model from ultralytics
    detection_model = YOLO(f"yolo11{args.yolo_model_size}.pt") #, device = video_predictor.device)  # model for segmantion and detection

    yolo_detections_output = []
    image_decomposition_positions = ["left", "front", "right", "back"]

    # Detect persons in images using YOLO
    for image_idx in tqdm(range(0, len(images_list), args.yolo_batch_size), desc="Processing input images (detection with YOLO)"):
        
        image_files_batch = images_list[image_idx:image_idx+args.yolo_batch_size]
        
        # image_file = image_input_files[image_idx]
        # batch_image_original_idx = []
        # batch_image_position = []
        batch_list = []
        original_images_list = []
        
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
            
            # batch_image_original_idx.extend([image_idx] * 4)
            # batch_image_position.extend(["left", "front", "right", "back"])
            batch_list.extend([image_left, image_front, image_right, image_back])
            
            if False:
                # FOR DEBUGGING
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
                stacked = cv2.resize(stacked, (stacked.shape[1]//8, stacked.shape[0]//8))
                stacked = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
                cv2.imshow("stacked", stacked)
                key = cv2.waitKey(int(1000/15))
                if key == ord('q') or key == 27:
                    exit(0)
                batch_list.append(stacked)
            
        batch_detections = detection_model(batch_list, verbose=False, device = video_predictor.device) # results of len args.yolo_batch_size * 4
        batch_detections = [batch_detections[i:i+4] for i in range(0, len(batch_detections), 4)] # list of list
        
        
        for batch_image_idx, subimages_detections in enumerate(batch_detections):

            original_image = original_images_list[batch_image_idx].copy()
            original_image_width = original_image.shape[1]
            subimage_width = original_image_width // 2

            current_image_detections = {"image_file": os.path.basename(image_files_batch[batch_image_idx]), "detections": []}
            
            for position, result in zip(image_decomposition_positions, subimages_detections):
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
                
                for box_idx, (box, class_name, confidence) in enumerate(zip(boxes, classes_names, confidences)):
                    if class_name != "person":
                        continue
                    
                    if confidence.item() < args.yolo_confidence_threshold:
                        continue
                    
                    xmin = box[0].item()
                    ymin = box[1].item()
                    xmax = box[2].item()
                    ymax = box[3].item()
                    if xmin < subimage_width//4 or xmin > 3 * (subimage_width//4):
                        # print(f"Edge detection for {position} : {int(xmin)} (allowed range : {subimage_width//4} - {3 * (subimage_width//4)})")
                        # do not consider "edge detections" they will be processed in another camera
                        continue
                    else:
                        pass
                        # print(f"Good detection for {position} : {int(xmin)} (allowed range : {subimage_width//4} - {3 * (subimage_width//4)})")
                    
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
                    
                    current_image_detections["detections"].append({"xmin": int(xmin),
                                                                    "ymin": int(ymin),
                                                                    "xmax": int(xmax),
                                                                    "ymax": int(ymax),
                                                                    "class_name": class_name,
                                                                    "confidence": confidence.item(),
                                                                    "validity": "valid" if confidence.item() > args.yolo_confidence_threshold else "low_conf"})

            yolo_detections_output.append(current_image_detections)

            if False:
                # draw box
                original_image_show = original_image.copy()
                for detection in current_image_detections["detections"]:
                    cv2.rectangle(original_image_show, (int(detection["xmin"]), int(detection["ymin"])), (int(detection["xmax"]), int(detection["ymax"])), (0, 0, 255), 2)
                    cv2.putText(original_image_show, detection["class_name"], (int(detection["xmin"]), int(detection["ymin"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
                img_show = original_image_show.copy()
                img_show = cv2.resize(img_show, (1920, 960))
                img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
                cv2.imshow("image_result", img_show)
                key = cv2.waitKey(0)
                if key == ord('q') or key == 27:
                    exit(0)
                cv2.destroyAllWindows()
    
    assert(len(yolo_detections_output) == len(images_list)), "Number of detections and images should be the same"
    
    video_predictor_initialized = False
    sam2_detections_output = {} # dict in the form {idx: {"image_file": FILENAME, "detections": [...]}, "idx": ...}
    current_sam2_track_id = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image_idx, image_file in tqdm(enumerate(images_list), desc="Processing input images (tracking with SAM2)"):
            
            prDebug("========= IMAGE {} ==========".format(image_idx))
            
            yolo_detections_in_image = yolo_detections_output[image_idx]["detections"]
            if len(yolo_detections_in_image) == 0:
                prWarning("No detections in image : {}, skipping".format(image_file))
                continue
            
            for yolo_detection_idx, yolo_detection in enumerate(yolo_detections_in_image):
                
                prDebug("===== YOLO DETECTION {} ======".format(yolo_detection_idx))
                
                # Create the entry in the sam2_detections_output if it does not exist
                if image_idx not in sam2_detections_output:
                    sam2_detections_output[image_idx] = {"image_file": image_file, "detections": []}
                
                image = Image.open(image_file)
                image = np.array(image) # [H, W, 3] with usually H = 1920, H = 3840 for 360 images
                yolo_detection_box = [yolo_detection["xmin"], yolo_detection["ymin"], yolo_detection["xmax"], yolo_detection["ymax"]]
                image_centered, box_in_image_centered, centered_image_xmin, centered_image_xmax = get_image_centered_on_detection(image, yolo_detection_box) # either projection or just cropping
                
                if video_predictor_initialized:
                    prInfo("Resetting video predictor")
                    video_predictor.reset_state()
                video_predictor.load_first_frame(image_centered)
                video_predictor_initialized = True
                
                # always pretend we are at ID 0 and frame 0 since we restart everything 
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_prompt(frame_idx=0, obj_id=0, bbox=box_in_image_centered)
                centered_image_mask = (out_mask_logits[0, 0, :, :] > 0.0)
                centered_image_mask = erode_and_dilate_mask(centered_image_mask, erode_and_dilate_n_pixels=11)
                bbox_in_centered_image = get_bounding_box_torch(centered_image_mask)
                bbox_in_absolute_image = [bbox_in_centered_image[0] + centered_image_xmin, 
                                          bbox_in_centered_image[1], 
                                          bbox_in_centered_image[2] + centered_image_xmin, 
                                          bbox_in_centered_image[3]] # this box can have negative coordinates or above the image shape
                
                full_image_mask = get_full_image_mask(centered_image_mask, centered_image_xmin, image.shape)
                full_image_mask_rle = encode_RLE(full_image_mask)
                
                
                matching_sam2_track = None
                matching_sam2_track_detection_idx = None
                for detection_comp_idx, detection_comp in enumerate(sam2_detections_output[image_idx]["detections"]):
                    detection_comp_box = [detection_comp["xmin"], detection_comp["ymin"], detection_comp["xmax"], detection_comp["ymax"]]
                    iou_boxes = bb_intersection_over_union_with_shift(bbox_in_absolute_image, detection_comp_box, shift_test=image.shape[1])
                    # prDebug(f"Comparing the detection generated from YOLO idx {yolo_detection_idx} with the SAM2 detection idx {detection_comp_idx} in the image {image_idx} => IoU = {np.round(iou, 2)}")
                    if iou_boxes > MIN_OVERLAP_IOU_BOXES:
                        detection_comp_mask_rle = detection_comp["mask_rle"]
                        detection_comp_mask = decode_RLE(detection_comp_mask_rle, shape = image.shape[:2])
                        iou_masks = get_iou_torch(detection_comp_mask, full_image_mask)
                        prop_of_mask_in_mask_comp = prop_of_mask1_in_mask2_torch(full_image_mask, detection_comp_mask)
                        if iou_masks > MIN_OVERLAP_IOU_MASKS or prop_of_mask_in_mask_comp > MIN_MASK_IN_MASK_COMP_PROP:
                            matching_sam2_track = detection_comp["track_id"]
                            matching_sam2_track_detection_idx = detection_comp_idx
                            break
                
                if matching_sam2_track is not None:
                    prWarning(f"Matching SAM2 track {matching_sam2_track} (idx {matching_sam2_track_detection_idx}) found for detection : {yolo_detection_idx}, do not pursue tracking from this detection (iou_boxes = {iou_boxes * 100:.2f}%, iou_masks = {iou_masks * 100:.2f}%, prop_of_mask_in_mask_comp = {prop_of_mask_in_mask_comp * 100:.2f}%)")
                    continue
                else:
                    prSuccess(f"No matching SAM2 track found for detection : {yolo_detection_idx}, pursuing tracking from this detection starting at image {image_idx}. Assigned SAM2 track ID {current_sam2_track_id}")
                    
                # save the detection for the current sam2 track...
                sam2_detections_output[image_idx]["detections"].append({"track_id": current_sam2_track_id,
                                                                        "xmin": bbox_in_absolute_image[0],
                                                                        "ymin": bbox_in_absolute_image[1],
                                                                        "xmax": bbox_in_absolute_image[2],
                                                                        "ymax": bbox_in_absolute_image[3],
                                                                        "mask_rle": full_image_mask_rle})
                
                # ... then iterate on the track and keep checking if it does not match with any existing detection
                empty_mask_count = 0
                for following_image_idx in range(image_idx+1, len(images_list)):
                    following_image_file = images_list[following_image_idx]
                    following_image = Image.open(following_image_file)
                    following_image = np.array(following_image)
                    # center based on the bounding box of the previous detection
                    following_image_centered, following_box_in_image_centered, following_image_xmin, following_image_xmax = get_image_centered_on_detection(following_image, bbox_in_absolute_image)
                    
                    out_obj_ids, out_mask_logits = video_predictor.track(following_image_centered)
                    centered_image_mask = (out_mask_logits[0, 0, :, :] > 0.0)
                    centered_image_mask = erode_and_dilate_mask(centered_image_mask, erode_and_dilate_n_pixels=11)
                    bbox_in_centered_image = get_bounding_box_torch(centered_image_mask)
                    if bbox_in_centered_image[0] is None:
                        empty_mask_count += 1
                        if empty_mask_count > MAX_OCCLUSION_FRAMES_COUNT:
                            empty_mask_count = 0
                            prWarning(f"Empty mask for SAM2 track id {current_sam2_track_id} at image {following_image_idx}, stop the track")
                            break
                        else:
                            prWarning(f"Empty mask for SAM2 track id {current_sam2_track_id} at image {following_image_idx}, skipping and continuing the track")
                            continue
                    else:
                        empty_mask_count = 0
                        
                    bbox_in_absolute_image = [bbox_in_centered_image[0] + following_image_xmin, 
                                              bbox_in_centered_image[1], 
                                              bbox_in_centered_image[2] + following_image_xmin, 
                                              bbox_in_centered_image[3]] # this box can have negative coordinates or above the image shape
                    full_image_mask = get_full_image_mask(centered_image_mask, following_image_xmin, following_image.shape)
                    full_image_mask_rle = encode_RLE(full_image_mask)
                    
                    if following_image_idx not in sam2_detections_output:
                        sam2_detections_output[following_image_idx] = {"image_file": following_image_file, "detections": []}
                    
                    # check if the detection matches with any existing SAM2 track
                    matching_sam2_track = None
                    matching_sam2_track_detection_idx = None
                    for detection_comp_idx, detection_comp in enumerate(sam2_detections_output[following_image_idx]["detections"]):
                        detection_comp_box = [detection_comp["xmin"], detection_comp["ymin"], detection_comp["xmax"], detection_comp["ymax"]]
                        iou_boxes = bb_intersection_over_union_with_shift(bbox_in_absolute_image, detection_comp_box, shift_test=following_image.shape[1])
                        # prDebug(f"Comparing the detection generated from SAM2 track id {current_sam2_track_id} with the SAM2 detection idx {detection_comp_idx} in the image {following_image_idx} => IoU = {np.round(iou, 2)}")
                        if iou_boxes > MIN_OVERLAP_IOU_BOXES:
                            detection_comp_mask_rle = detection_comp["mask_rle"]
                            detection_comp_mask = decode_RLE(detection_comp_mask_rle, shape = following_image.shape[:2])
                            iou_masks = get_iou_torch(detection_comp_mask, full_image_mask)
                            prop_of_mask_in_mask_comp = prop_of_mask1_in_mask2_torch(full_image_mask, detection_comp_mask) # if we are fully enclosed in the compared mask, we break
                            if iou_masks > MIN_OVERLAP_IOU_MASKS or prop_of_mask_in_mask_comp > MIN_MASK_IN_MASK_COMP_PROP:
                                matching_sam2_track = detection_comp["track_id"]
                                matching_sam2_track_detection_idx = detection_comp_idx
                                break
                    
                    if matching_sam2_track is not None:
                        prWarning(f"Matching SAM2 track {matching_sam2_track} (idx {matching_sam2_track_detection_idx}) found while tracking with SAM2 track id {current_sam2_track_id} at image {following_image_idx}, do not pursue tracking from this detection (iou_boxes = {iou_boxes * 100:.2f}%, iou_masks = {iou_masks * 100:.2f}%, prop_of_mask_in_mask_comp = {prop_of_mask_in_mask_comp * 100:.2f}%)")
                        break
         
                    # save the detection for the current sam2 track...
                    sam2_detections_output[following_image_idx]["detections"].append({"track_id": current_sam2_track_id,
                                                                                        "xmin": bbox_in_absolute_image[0],
                                                                                        "ymin": bbox_in_absolute_image[1],
                                                                                        "xmax": bbox_in_absolute_image[2],
                                                                                        "ymax": bbox_in_absolute_image[3],
                                                                                        "mask_rle": full_image_mask_rle})
                    
                    prDebug(f"[{following_image_idx}][{current_sam2_track_id}] Box : [{int(bbox_in_absolute_image[0])}, {int(bbox_in_absolute_image[1])}, {int(bbox_in_absolute_image[2])}, {int(bbox_in_absolute_image[3])}]")
                    prDebug(f"Mask shape : {centered_image_mask.shape}, image centered shape : {following_image_centered.shape}")
                    
                    
                    # draw
                    image_centered_show = following_image_centered.copy()
                    mask_image = image_centered_show.copy()
                    mask_image[centered_image_mask.cpu().numpy()] = COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)]
                    image_centered_show = cv2.addWeighted(image_centered_show, 0.5, mask_image, 0.5, 0)
                    # image_centered_show[centered_image_mask.cpu().numpy()] = [0, 0, 255]
                    cv2.putText(image_centered_show, f"ID {current_sam2_track_id} (image {following_image_idx})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                    cv2.rectangle(image_centered_show, (int(bbox_in_centered_image[0]), int(bbox_in_centered_image[1])), (int(bbox_in_centered_image[2]), int(bbox_in_centered_image[3])), COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                    image_centered_show = cv2.cvtColor(image_centered_show, cv2.COLOR_BGR2RGB)
                    image_centered_show = cv2.resize(image_centered_show, (480, 480))
                    # image_centered_show = cv2.drawContours(image_centered_show, [mask.cpu().numpy().astype(np.uint8)], 0, (0, 0, 255), 2)
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
                    # image_full_show[full_image_mask.cpu().numpy()] = [0, 0, 255]
                    cv2.putText(image_full_show, f"ID {current_sam2_track_id} (image {following_image_idx})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                    cv2.rectangle(image_full_show, (int(bbox_in_absolute_image[0]), int(bbox_in_absolute_image[1])), (int(bbox_in_absolute_image[2]), int(bbox_in_absolute_image[3])), COLORS_SEQUENCE_BGR[current_sam2_track_id % len(COLORS_SEQUENCE_BGR)], 2)
                    image_full_show = cv2.cvtColor(image_full_show, cv2.COLOR_BGR2RGB)
                                        
                    track_save_debug_dir = os.path.join(args.dir, "track_debug", "track_{:06d}".format(current_sam2_track_id))
                    if not os.path.isdir(track_save_debug_dir):
                        os.makedirs(track_save_debug_dir)
                    cv2.imwrite(os.path.join(track_save_debug_dir, f"image_full_{following_image_idx:06d}.jpg"), image_full_show)
                    
                    image_full_show = cv2.resize(image_full_show, (960, 480))
                    cv2.imshow("image_full_show", image_full_show)
                    key = cv2.waitKey(10)
                    if key == ord('q') or key == 27:
                        exit(0)
                                    
                current_sam2_track_id += 1
    
    sam2_detections_output_list = []
    for image_idx, image_detections in sam2_detections_output.items():
        for det in image_detections["detections"]:
            det["mask_rle"] = det["mask_rle"].cpu().numpy().tolist()
        sam2_detections_output_list.append(image_detections)
    
    write_dic_to_json_file(sam2_detections_output_list, os.path.join(args.dir, "sam2_detections_360.json"))
    
    assert(len(sam2_detections_output_list) == len(images_list)), "Number of detections and images should be the same"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./demo/output/2025_06_13_11_15_31_Theta360")
    parser.add_argument("--sam2_model", "-sm", type=str, default="tiny", help="SAM2 model to use (tiny, small, base, large)")
    parser.add_argument("--yolo_model_size", "-ys", type=str, default="x", help="YOLO model size to use (n, s, l, x)")
    parser.add_argument("--yolo_confidence_threshold", "-yct", type=float, default=0.7, help="YOLO confidence threshold")
    # parser.add_argument("--erode_masks", "-em", type=int, default=11, help="Erode masks by this number of pixels")
    parser.add_argument("--yolo_batch_size", "-ybs", type=int, default=16, help="Batch size for the YOLO detection")
    parser.add_argument("--vos_optimized", "-vos", action="store_true", default=False, help="Use the VOS optimized model")
    # parser.add_argument("--ngpu", "-ngpu", type=int, default=0, help="Number of GPUs to use, default 0 means use all GPUs")
    args = parser.parse_args()

    # assert(args.erode_masks % 2 == 1), "Erode masks must be odd"
    
    prInfo("Using SAM2 model : {}".format(args.sam2_model))
    prInfo("Using YOLO model size : {}".format(args.yolo_model_size))
    
    images_path = os.path.join(args.dir, "images_360")
    assert os.path.isdir(images_path), "Directory should exist : {}".format(images_path)

    sam2_checkpoint = SAM2_MODEL_TO_CHECKPOINT[args.sam2_model]
    sam2_cfg = SAM2_MODEL_TO_CFG[args.sam2_model]
    
    assert os.path.isfile(sam2_checkpoint), "Checkpoint should exist : {}".format(sam2_checkpoint)
    assert os.path.isfile(os.path.join(here, "sam2",sam2_cfg)), "SAM2 config should exist : {}".format(sam2_cfg)

    video_predictor = build_sam2_camera_predictor(sam2_cfg, sam2_checkpoint, device=torch.device("cuda", index=0), vos_optimized=args.vos_optimized)
    prSuccess("SAM2 model built successfully!")
    main(args, video_predictor)
