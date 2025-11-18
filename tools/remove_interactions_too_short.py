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
import datetime
import copy



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

    for episode in episodes_list:
        detection_file = os.path.join(args.dir, "episodes", episode, "sam2_detections_360.json")
        assert os.path.isfile(detection_file), f"Detection file does not exist at {detection_file}"
        detections_dict = read_json_to_dic(detection_file)
        
        images_dir = os.path.join(args.dir, "episodes", episode, "images_360")
        assert os.path.isdir(images_dir), f"Images directory does not exist at {images_dir}"
        images_list = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]
        images_list.sort()
        
        
        assert(len(detections_dict) == len(images_list)), f"Number of detections and images should be the same (got {len(detections_dict)} detections and {len(images_list)} images)"
        
        
        # first pass create segments
        detection_interact_segments_by_track_id = {}
        
        for image_idx, image_detections in tqdm(enumerate(detections_dict), total=len(detections_dict), desc="Creating segments"):
            # image_file = image_detections["image_file"]
            # image_file_from_list = images_list[image_idx]
            for detection in image_detections["detections"]:
                track_id = detection["track_id"]
                if track_id not in detection_interact_segments_by_track_id:
                    detection_interact_segments_by_track_id[track_id] = np.zeros(len(images_list), dtype=bool)
                
                engagement_level = detection.get("engagement_level", -1)
                if engagement_level == 3:
                    detection_interact_segments_by_track_id[track_id][image_idx] = True

        
        # second pass remove segments too short (erode by 1)
        eroded_detection_interact_segments_by_track_id = copy.deepcopy(detection_interact_segments_by_track_id)

        for image_idx, image_detections in tqdm(enumerate(detections_dict), total=len(detections_dict), desc="Removing segments too short"):
            # image_file = image_detections["image_file"]
            # image_file_from_list = images_list[image_idx]
            for detection in image_detections["detections"]:
                track_id = detection["track_id"]
                if detection_interact_segments_by_track_id[track_id][image_idx]:
                    if (detection_interact_segments_by_track_id[track_id][max(0, image_idx - 1)] == False) or (detection_interact_segments_by_track_id[track_id][min(len(images_list) - 1, image_idx + 1)] == False):
                        eroded_detection_interact_segments_by_track_id[track_id][image_idx] = False
                        
        # third pass dilate by 1
        dilated_detection_interact_segments_by_track_id = copy.deepcopy(eroded_detection_interact_segments_by_track_id)
        for image_idx, image_detections in tqdm(enumerate(detections_dict), total=len(detections_dict), desc="Dilating segments"):
            # image_file = image_detections["image_file"]
            # image_file_from_list = images_list[image_idx]
            for detection in image_detections["detections"]:
                track_id = detection["track_id"]
                if eroded_detection_interact_segments_by_track_id[track_id][image_idx]:
                    dilated_detection_interact_segments_by_track_id[track_id][max(0, image_idx - 1)] = True
                    dilated_detection_interact_segments_by_track_id[track_id][min(len(images_list) - 1, image_idx + 1)] = True
                    
        # fourth pass replace in detections dict
        for image_idx, image_detections in tqdm(enumerate(detections_dict), total=len(detections_dict), desc="Replacing in detections dict"):
            for detection in image_detections["detections"]:
                track_id = detection["track_id"]
                if dilated_detection_interact_segments_by_track_id[track_id][image_idx]:
                    detection["engagement_level"] = 3
                else:
                    detection["engagement_level"] = 0
                    
        # copy the detections dict to a new file for backup
        current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        backup_detection_file = detection_file.replace(".json", f"_copy_{current_timestamp}.json")
        shutil.copy(detection_file, backup_detection_file)
        prInfo(f"Copied detections to {backup_detection_file} for backup")
                    
        # write the detections dict to the file
        prInfo(f"Saving detections to {detection_file} with removed interactions too short")
        write_dic_to_json_file(detections_dict, detection_file)