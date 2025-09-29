import argparse
import os
import sys
import datetime
import numpy as np
import cv2
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *
from misc.other_utils import *
from misc.rle_tools import decode_RLE


MIN_PIXELS_INTERSECTION = 10000


def main(args, episode: str):
    """ Main function to get the engagement levels for the tracks

    Args:
        args (argparse.Namespace): arguments from the command line
        episode (str): episode to process
    """
    
    interact_zone_tracking_path = os.path.join(args.dir, "episodes", episode, "interact_zone_tracking_360.json")
    if os.path.exists(interact_zone_tracking_path):
        prInfo(f"Interact zone tracking file {interact_zone_tracking_path} exists using it to get a different mask for each frame. Ignoring the default or specified mask ({args.intersection_mask_path})")
        interact_zone_tracking_dict = read_json_to_dic(interact_zone_tracking_path) # list of detection of type : [{"image_file": "image_file.jpg", "detections": [{"track_id": 1, "xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100, "mask_rle": "mask_rle"}, ...]}, ...]
        intersection_mask = None
    else:
        prInfo(f"Interact zone tracking file {interact_zone_tracking_path} does not exist using the default or specified mask ({args.intersection_mask_path}) for all frames")
        interact_zone_tracking_dict = None
        intersection_mask = cv2.imread(args.intersection_mask_path)
        intersection_mask = cv2.cvtColor(intersection_mask, cv2.COLOR_BGR2GRAY)
        intersection_mask = np.array(intersection_mask>128)
        intersection_mask = torch.from_numpy(intersection_mask) # Tensor of shape [H, W], dtype is torch.bool
    
    image_dir = os.path.join(args.dir, "episodes", episode, "images_360")
    image_files = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg")]
    image_files.sort()
    
    width, height = cv2.imread(image_files[0]).shape[:2]
    
    detections_path = os.path.join(args.dir, "episodes", episode, "sam2_detections_360.json")
    if not os.path.exists(detections_path):
        raise ValueError("Detection file {} does not exist".format(detections_path))
    
    detections_dict = read_json_to_dic(detections_path) # list of detection of type : [{"image_file": "image_file.jpg", "detections": [{"track_id": 1, "xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100, "mask_rle": "mask_rle"}, ...]}, ...]

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    detections_path_save = detections_path.replace(".json", f"_copy_{timestamp}.json")
    write_dic_to_json_file(detections_dict, detections_path_save)
    prInfo(f"Saved detections copy to {detections_path_save}")

    for image_idx, image_detections in tqdm(enumerate(detections_dict), total=len(detections_dict), desc="Processing detections"):
        for det_idx, det in enumerate(image_detections["detections"]):
            mask_rle = np.array(det["mask_rle"]).astype(np.int32)
            mask = decode_RLE(torch.from_numpy(mask_rle), shape = (height, width))
            if intersection_mask is not None:
                intersection_sum = torch.sum(intersection_mask & mask)
            else:
                compared_mask_rle = np.array(interact_zone_tracking_dict[image_idx]["detections"][0]["mask_rle"]).astype(np.int32)
                compared_mask = decode_RLE(torch.from_numpy(compared_mask_rle), shape = (height, width))
                intersection_sum = torch.sum(compared_mask & mask)
                
            if intersection_sum > MIN_PIXELS_INTERSECTION:
                det["engagement_level"] = 3
            else:
                det["engagement_level"] = 0
                
    # write the detections dict to the file
    prInfo(f"Saving detections to {detections_path}")
    write_dic_to_json_file(detections_dict, detections_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./demo/output/2025_06_13_11_15_31_Theta360")
    parser.add_argument("--episodes", "-e", type=str, default="all", help="Episodes to use, separated by commas or 'all'")
    parser.add_argument("--intersection_mask_path", "-imp", type=str, default="./assets/default_mask_intersection_zone.png")
    args = parser.parse_args()
    
    assert os.path.isdir(args.dir), "Directory should exist at {}".format(args.dir)
    assert os.path.isfile(args.intersection_mask_path), "Intersection mask path should exist at {}".format(args.intersection_mask_path)

    # List of episodes to process
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()
    
    prInfo("Auto labelling : {}".format(os.path.basename(args.dir)))
    
    for episode in episodes_list:
        main(args, episode)