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

RECORDINGS = [
    "2023_07_06_albee_square_landfill_0",
    "2023_07_06_albee_square_landfill_1",
    "2023_07_07_albee_square_landfill_0",
    "2023_07_07_albee_square_landfill_1",
    "2023_07_11_albee_square_landfill_0",
    "2023_07_11_albee_square_landfill_1",
    "2023_07_12_albee_square_landfill_0",
    # "2023_07_12_albee_square_landfill_1",
    "2023_07_14_albee_square_landfill",
    "2023_07_06_albee_square_recycle_0",
    "2023_07_06_albee_square_recycle_1",
    "2023_07_07_albee_square_recycle_0",
    "2023_07_07_albee_square_recycle_1",
    "2023_07_11_albee_square_recycle_0",
    "2023_07_11_albee_square_recycle_1",
    "2023_07_12_albee_square_recycle_0",
    "2023_07_12_albee_square_recycle_1",
    "2023_07_14_albee_square_recycle"
    ]

RECORDINGS_DIR = "/home/raphael/group_storage/rlorenzo/SSUP/FullVideos/extracted"

# Global variables for mouse callback
click_points = []
display_image = None

def mouse_callback(event, x, y, flags, param):
    global click_points, display_image
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        # Draw a circle at the click point
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("First image", display_image)
        #print(f"Clicked at: ({x}, {y})")

for recording in RECORDINGS:
    recording_dir = os.path.join(RECORDINGS_DIR, recording)
    assert os.path.isdir(recording_dir), f"Recording directory does not exist: {recording_dir}"
    images_dir = os.path.join(recording_dir, "episodes", "0000", "images_360")
    assert os.path.isdir(images_dir), f"Images directory does not exist: {images_dir}"
    images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_dir)) for f in fn if f.endswith(".jpg")]
    images.sort()
    
    first_image_path = images[0]
    
    first_image = cv2.imread(first_image_path)
    original_height, original_width = first_image.shape[:2]
    
    # Resize image to 50% (divide by 2)
    resized_height = original_height // 2
    resized_width = original_width // 2
    resized_image = cv2.resize(first_image, (resized_width, resized_height))
    
    cv2.namedWindow("First image")
    cv2.setMouseCallback("First image", mouse_callback)
    
    confirmed = False
    while not confirmed:
        # Reset click points for this attempt
        click_points = []
        display_image = resized_image.copy()
        cv2.imshow("First image", display_image)
        
        # print(f"Click on the topleft and topright corner of the trashcan for {recording}")
        
        # Wait for two clicks
        while len(click_points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                exit(0)
        
        # Get the two click points
        pt1, pt2 = click_points[0], click_points[1]
        
        # Calculate bbox in resized coordinates
        xmin_resized = min(pt1[0], pt2[0])
        ymin_resized = min(pt1[1], pt2[1])
        xmax_resized = max(pt1[0], pt2[0])
        ymax_resized = max(pt1[1], pt2[1])
        
        # Draw the bounding box on the display image
        cv2.rectangle(display_image, (xmin_resized, ymin_resized), (xmax_resized, ymax_resized), (0, 255, 0), 2)
        cv2.imshow("First image", display_image)
        
        # Convert to original image coordinates (multiply by 2)
        xmin = xmin_resized * 2
        ymin = ymin_resized * 2
        xmax = xmax_resized * 2
        ymax = ymax_resized * 2
        
        # Ask for confirmation
        # print("Press Y or Enter to confirm, N to restart")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                exit(0)
            elif key == ord('y') or key == ord('Y') or key == 13:  # Y or Enter
                # Confirm and print the result
                print(f"{recording} : {xmin},{ymin},{xmax},{ymax}")
                confirmed = True
                break
            elif key == ord('n') or key == ord('N'):
                # Restart: remove bounding box by resetting display_image
                # print("Restarting...")
                break
    #cv2.destroyAllWindows()