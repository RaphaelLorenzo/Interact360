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
        
        for image_idx, image_detections in enumerate(detections_dict):
            image_file = image_detections["image_file"]
            image_file_from_list = images_list[image_idx]
            
            # print(f"Image file: {image_file} vs image file from list: {image_file_from_list}")
            pass
            
            # Dont do anything for now