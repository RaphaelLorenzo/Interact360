import torch
import numpy as np
import argparse
import os
import sys
import datetime
import torch.multiprocessing as mp
from typing import Optional
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *
from misc.other_utils import *

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

def process_episode_worker(args, ep, image_processor):
    """Worker function that loads the model and processes a single episode"""
    device = torch.device("cuda")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", cache_dir=os.path.join(here, "..", "checkpoints"))
    model.to(device)
    
    return process_episode(args, ep, image_processor, model, device)

def process_episode(args, ep, image_processor, model, device):

    image_dir = os.path.join(args.dir, "episodes", ep, "images_360")
    image_files = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg")]
    image_files.sort()
    
    detections_path = os.path.join(args.dir, "episodes", ep, "sam2_detections_360.json")
    if not os.path.exists(detections_path):
        raise ValueError("Detection file {} does not exist".format(detections_path))
    
    detections_dict = read_json_to_dic(detections_path)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    detections_path_save = detections_path.replace(".json", f"_copy_{timestamp}.json")
    write_dic_to_json_file(detections_dict, detections_path_save)
    prInfo(f"Saved detections copy to {detections_path_save}")
    
    for image_idx in tqdm(range(len(detections_dict)), total=len(detections_dict), desc="Extracting poses"):
        
        image_file = os.path.basename(detections_dict[image_idx]["image_file"]) # basename of the image
        image_path = os.path.join(args.dir, "episodes", ep, "images_360", image_file)
        assert os.path.exists(image_path), f"Image path {image_path} does not exist"
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        detections = detections_dict[image_idx]["detections"]

        person_boxes = [] # list of list such that it is interepreted as a batch of one box
        valid_boxes_idxs = []
        images_centered_on_valid_detections = []
        images_xmins_centered_on_valid_detections = []
        invalid_boxes_idxs = []
        for idx, det in enumerate(detections):
            box = [det["xmin"], det["ymin"], det["xmax"], det["ymax"]]
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]] # convert to COCO format
            
            if box[2] > 50.0 and box[3] > 100.0:
                # box big enough
                valid_boxes_idxs.append(idx)
                # create an image centered on the detection
                image_centered, box_in_image_centered, centered_image_xmin, _ = get_image_centered_on_detection(np.array(image), [det["xmin"], det["ymin"], det["xmax"], det["ymax"]])
                images_centered_on_valid_detections.append(image_centered)
                images_xmins_centered_on_valid_detections.append(centered_image_xmin)
                box_in_image_centered_coco = [box_in_image_centered[0], box_in_image_centered[1], box_in_image_centered[2] - box_in_image_centered[0], box_in_image_centered[3] - box_in_image_centered[1]]
                person_boxes.append([np.array(box_in_image_centered_coco)]) # list of list such that it is interepreted as a batch of one box
                
            else:
                invalid_boxes_idxs.append(idx)

        if len(valid_boxes_idxs) > 0:
            
            for batch_idx in range(0, len(valid_boxes_idxs), args.batch_size):
                
                images_centered_on_valid_detections_batch = images_centered_on_valid_detections[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                person_boxes_batch = person_boxes[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                images_xmins_centered_on_valid_detections_batch = images_xmins_centered_on_valid_detections[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                valid_boxes_idxs_batch = valid_boxes_idxs[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                
                inputs = image_processor(images_centered_on_valid_detections_batch, boxes=person_boxes_batch, return_tensors="pt").to(device) # NOTE : this does not allow to put the masks on the images / Must be sure that the box contains only one person

                with torch.no_grad():
                    outputs = model(**inputs)

                pose_results = image_processor.post_process_pose_estimation(outputs, boxes=person_boxes_batch)
                del inputs, outputs
                torch.cuda.empty_cache()
                            
                for pose, detection_idx, image_xmin in zip(pose_results, valid_boxes_idxs_batch, images_xmins_centered_on_valid_detections_batch):
                    keypoints = pose[0]["keypoints"].cpu().numpy() # shape (NKP, 2)
                    keypoints = get_full_image_keypoints(keypoints, image_xmin, (image_height, image_width))
                    keypoints = keypoints.flatten().tolist()
                    keypoints = [int(coord) for coord in keypoints]
                    
                    scores = pose[0]["scores"].cpu().numpy().flatten().tolist()
                    scores = [np.round(float(score), 2) for score in scores]
                    
                    detections_dict[image_idx]["detections"][detection_idx]["keypoints_vitpose"] = keypoints
                    detections_dict[image_idx]["detections"][detection_idx]["scores_vitpose"] = scores
            
        for idx in invalid_boxes_idxs:
            detections_dict[image_idx]["detections"][idx]["keypoints_vitpose"] = [-1] * 17 * 2 # 17 keypoints, 2 coordinates
            detections_dict[image_idx]["detections"][idx]["scores_vitpose"] = [0.0] * 17 # 17 scores
    
    write_dic_to_json_file(detections_dict, detections_path)
    prInfo(f"Saved detections to {detections_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./output/rosbags_2025_03_26_15_25_25")
    parser.add_argument("--episodes", "-e", type=str, default="all")
    parser.add_argument("--workers", "-w", type=int, default=0)
    parser.add_argument("--batch_size", "-b", type=int, default=16)

    args = parser.parse_args()
    
    assert os.path.isdir(args.dir), "Directory should exist"
    assert os.path.isdir(os.path.join(args.dir, "episodes")), "Episodes directory should exist"
    
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()

    for ep in episodes_list:
        assert os.path.isdir(os.path.join(args.dir, "episodes", ep, "images_360")), "Episode {} should exist".format(ep)
        assert os.path.isfile(os.path.join(args.dir, "episodes", ep, "sam2_detections_360.json")), "Detection file {} does not exist".format(os.path.join(args.dir, "episodes", ep, "sam2_detections_360.json"))

    prInfo("Extracting poses for episodes : {}".format(", ".join(episodes_list)))
    
    if args.workers == 0:
        # get total cuda memory
        device = torch.device("cuda")
        total_cuda_memory = torch.cuda.get_device_properties(device).total_memory
        prInfo(f"Total CUDA memory: {total_cuda_memory / 1024**3:.2f} GB")
        num_workers = total_cuda_memory // (2*(1024**3)) # keep 2GB for each worker
        num_workers = min(num_workers, len(episodes_list))
        total_cpu_count = mp.cpu_count()
        num_workers = min(num_workers, total_cpu_count-1)
        prInfo(f"Using {num_workers} workers")
    else:
        num_workers = args.workers
        prInfo(f"Using {num_workers} workers")
    
    prInfo("Extracting poses using {} workers to process {} episodes in parallel".format(num_workers, len(episodes_list)))

    image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    prInfo("Loaded image processor")
    
    # use torch multiprocessing to process the episodes in parallel
    with Pool(processes=num_workers) as pool:
        pool.starmap(process_episode_worker, [(args, ep, image_processor) for ep in episodes_list])
