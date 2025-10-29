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
sys.path.append(os.path.join(here, '../../'))
sys.path.append(os.path.join(here))

from sam2.build_sam import build_sam2_camera_predictor, build_sam2
from sam2.sam2_camera_predictor import SAM2CameraPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from transformers import AutoProcessor, VitPoseForPoseEstimation

from misc.print_utils import *
from misc.other_utils import *
from misc.rle_tools import encode_RLE, decode_RLE

import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./demo/output/2025_06_13_11_15_31_Theta360", help="Directory to use (default: %(default)s)")
    parser.add_argument("--episodes", "-e", type=str, default="all", help="Episodes to use, separated by commas or 'all' (default: %(default)s)")
    args = parser.parse_args()

    # List of episodes to process
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()

    first_ts = None

    for ep_idx, episode in enumerate(episodes_list):
        
        detection_file = os.path.join(args.dir, "episodes", episode, "sam2_detections_360.json")
        assert os.path.isfile(detection_file), f"Detection file does not exist at {detection_file}"
        detections_dict = read_json_to_dic(detection_file)
        
        images_dir = os.path.join(args.dir, "episodes", episode, "images_360")
        assert os.path.isdir(images_dir), f"Images directory does not exist at {images_dir}"
        images_list = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]
        images_list.sort()
        
        interact_zone_tracking_file = os.path.join(args.dir, "episodes", episode, "interact_zone_tracking_360.json")
        assert os.path.isfile(interact_zone_tracking_file), f"Interact zone tracking file does not exist at {interact_zone_tracking_file}"
        interact_zone_tracking_dict = read_json_to_dic(interact_zone_tracking_file)
        
        assert(len(detections_dict) == len(images_list)), f"Number of detections and images should be the same (got {len(detections_dict)} detections and {len(images_list)} images)"
        assert(len(detections_dict) == len(interact_zone_tracking_dict)), f"Number of detections and interact zone tracking should be the same (got {len(detections_dict)} detections and {len(interact_zone_tracking_dict)} interact zone tracking)"
        
        for image_idx in range(len(images_list)):
            
            if image_idx == 0 and ep_idx == 0:
                first_ts = datetime.datetime.strptime(os.path.basename(images_list[image_idx]).split(".")[0], "%Y_%m_%d_%H_%M_%S_%f").timestamp()
            
            assert(first_ts is not None), f"First timestamp is not set"
            
            image_file = os.path.basename(images_list[image_idx])
            image_file_from_interact_zone_tracking = os.path.basename(interact_zone_tracking_dict[image_idx]["image_file"])
            image_file_from_detections = os.path.basename(detections_dict[image_idx]["image_file"])
            
            ts_from_image_file = datetime.datetime.strptime(image_file.split(".")[0], "%Y_%m_%d_%H_%M_%S_%f").timestamp()
            ts_from_image_file_from_interact_zone_tracking  = datetime.datetime.strptime(image_file_from_interact_zone_tracking.split(".")[0], "%Y_%m_%d_%H_%M_%S_%f").timestamp()
            ts_from_image_file_from_detections  = datetime.datetime.strptime(image_file_from_detections.split(".")[0], "%Y_%m_%d_%H_%M_%S_%f").timestamp()
            
            assert(ts_from_image_file == ts_from_image_file_from_interact_zone_tracking), f"Timestamp from image file: {ts_from_image_file} vs timestamp from interact zone tracking: {ts_from_image_file_from_interact_zone_tracking}"
            assert(ts_from_image_file == ts_from_image_file_from_detections), f"Timestamp from image file: {ts_from_image_file} vs timestamp from detections: {ts_from_image_file_from_detections}"
            
            new_ts = ts_from_image_file - first_ts
            new_ts_str = datetime.datetime.fromtimestamp(new_ts).strftime("%Y_%m_%d_%H_%M_%S_%f")
            
            print(f"[{episode}][{image_idx}/{len(images_list)}] Timestamp from image file: {ts_from_image_file} -> {ts_from_image_file - first_ts} -> {new_ts_str}")
            print(f"[{episode}][{image_idx}/{len(images_list)}] Timestamp from interact zone tracking: {ts_from_image_file_from_interact_zone_tracking} -> {ts_from_image_file_from_interact_zone_tracking - first_ts} -> {new_ts_str}")
            print(f"[{episode}][{image_idx}/{len(images_list)}] Timestamp from detections: {ts_from_image_file_from_detections} -> {ts_from_image_file_from_detections - first_ts} -> {new_ts_str}")
            
            new_image_file_name = f"{new_ts_str}.jpg"
            new_image_file = os.path.join(images_dir, new_image_file_name)
            
            # rename the image file to the relative timestamp
            shutil.move(os.path.join(images_dir, image_file), new_image_file)
            detections_dict[image_idx]["image_file"] = new_image_file_name
            interact_zone_tracking_dict[image_idx]["image_file"] = new_image_file_name
            
            prSuccess(f"Renamed image file {image_file} to {new_image_file_name}")
            print("--------------------------------")
        
        # copy the previous detections and interact zone tracking dictionaries to json files
        current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        shutil.copy(os.path.join(args.dir, "episodes", episode, "sam2_detections_360.json"), os.path.join(args.dir, "episodes", episode, f"sam2_detections_360_copy_{current_timestamp}.json"))
        shutil.copy(os.path.join(args.dir, "episodes", episode, "interact_zone_tracking_360.json"), os.path.join(args.dir, "episodes", episode, f"interact_zone_tracking_360_copy_{current_timestamp}.json"))
        
        prSuccess(f"Copied previous detections and interact zone tracking dictionaries to json files")
        
        # write the detections and interact zone tracking dictionaries to json files
        write_dic_to_json_file(detections_dict, os.path.join(args.dir, "episodes", episode, "sam2_detections_360.json"))
        write_dic_to_json_file(interact_zone_tracking_dict, os.path.join(args.dir, "episodes", episode, "interact_zone_tracking_360.json"))
        
        prSuccess(f"Wrote detections and interact zone tracking dictionaries to json files with updated image file names")