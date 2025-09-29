import os
import sys
import subprocess
import shutil
import time
import multiprocessing as mp

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.geometry_utils import *
from misc.other_utils import *

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import argparse

PERSPECTIVE_VIEWS = [(0, 0), (0, 25), (75, 0), (75, 25), (150, 0), (150, 25), (225, 0), (225, 25), (300, 0), (300, 25)]

def main(args, episode: str):
    """ Main function to extract perspective images """
    
    image_dir = os.path.join(args.dir, "episodes", episode, "images_360")
    image_files = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg")]
    image_files.sort()
    
    if args.extract_at_idxs == "all":
        extract_at_idxs = list(range(len(image_files)))
    else:
        extract_at_idxs = [int(x) for x in args.extract_at_idxs.split(",")]
    
    for idx in extract_at_idxs:
        
        image = cv2.imread(image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.convert("RGB")
        image = np.array(image)
        
        if not os.path.exists(os.path.join(args.dir, "episodes", episode, "perspective_images")):
            os.makedirs(os.path.join(args.dir, "episodes", episode, "perspective_images"))
        
        for yaw, pitch in PERSPECTIVE_VIEWS:
            map_x, map_y = perspective_view_torch(image.shape[:2], args.fov, image.shape[1], image.shape[0], device="cuda", yaw=yaw, pitch=pitch, z = 2)
            # rev_map_x, rev_map_y = reverse_perspective_view_torch(image.shape[:2], args.fov, image.shape[1], image.shape[0], device="cuda", yaw=yaw, pitch=pitch)
            perspective_image = cv2.remap(image, map_x.cpu().numpy(), map_y.cpu().numpy(), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            perspective_image = cv2.cvtColor(perspective_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.dir, "episodes", episode, "perspective_images", f"{idx:06d}_yaw_{yaw}_pitch_{pitch}.jpg"), perspective_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./demo/output/2025_06_13_11_15_31_Theta360")
    parser.add_argument("--episodes", "-e", type=str, default="all", help="Episodes to use, separated by commas or 'all'")
    parser.add_argument("--fov", "-fov", type=float, default=90, help="Field of view in degrees")
    parser.add_argument("--extract_at_idxs", "-eai", type=str, default="100", help="Indexes to extract, separated by commas or 'all'")

    args = parser.parse_args()
    
    assert os.path.isdir(args.dir), "Directory should exist at {}".format(args.dir)

    # List of episodes to process
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()
        
    for episode in episodes_list:
        main(args, episode)