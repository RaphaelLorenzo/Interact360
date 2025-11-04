import argparse
import os
import sys
import datetime
import numpy as np
import cv2
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *
from misc.other_utils import *
from misc.rle_tools import decode_RLE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_one_image_element(args, intersection_mask, detections, height, width, interaction_zone_tracking_rle):
    if intersection_mask is None and len(detections) > 0:
        # load once for the image
        compared_mask_rle = torch.from_numpy(np.array(interaction_zone_tracking_rle.split(",")).astype(np.int32)).to(DEVICE)
        compared_mask = decode_RLE(compared_mask_rle, shape = (width, height))
        
    for det_idx, det in enumerate(detections):
        mask_rle = np.array(det["mask_rle"].split(",")).astype(np.int32)
        mask = decode_RLE(torch.from_numpy(mask_rle).to(DEVICE), shape = (width, height))
        # cv2.imshow(f"Mask {det['track_id']}", cv2.resize((mask*255).cpu().numpy().astype(np.uint8), (960, 480)))
        # key = cv2.waitKey(0)
        # if key == ord('q') or key == 27:
        #     exit()
        if intersection_mask is not None:
            intersection_sum = torch.sum(intersection_mask & mask)
        else:
            intersection_sum = torch.sum(compared_mask & mask)
            
        if intersection_sum > args.min_pixels_intersection:
            det["engagement_level"] = 3
        else:
            det["engagement_level"] = 0
        
        det["intersection_pixels"] = intersection_sum.item()
        
    return detections

def worker_process_image(args_tuple):
    """Worker function for multiprocessing that processes a single image element"""
    (image_idx, image_detections, intersection_mask, height, width, 
     interaction_zone_tracking_rle, min_pixels_intersection) = args_tuple
    
    # Create a minimal args object for the worker
    class WorkerArgs:
        def __init__(self, min_pixels_intersection):
            self.min_pixels_intersection = min_pixels_intersection
    
    worker_args = WorkerArgs(min_pixels_intersection)
    
    processed_detections = process_one_image_element(
        worker_args, intersection_mask, image_detections["detections"], 
        height, width, interaction_zone_tracking_rle
    )
    
    return image_idx, processed_detections

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
        intersection_mask = torch.from_numpy(intersection_mask).to(DEVICE) # Tensor of shape [H, W], dtype is torch.bool
    
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

    # Prepare arguments for multiprocessing
    process_args = []
    for image_idx, image_detections in enumerate(detections_dict):
        if interact_zone_tracking_dict is not None:
            interaction_zone_tracking_rle = interact_zone_tracking_dict[image_idx]["detections"][0]["mask_rle"]
        else:
            interaction_zone_tracking_rle = None
            
        process_args.append((
            image_idx, image_detections, intersection_mask, height, width,
            interaction_zone_tracking_rle, args.min_pixels_intersection
        ))
    
    # Use torch multiprocessing to process images in parallel
    num_processes = min(mp.cpu_count()//2, len(detections_dict)) if args.workers == 0 else args.workers
    num_processes = min(num_processes, len(process_args))
    prInfo(f"Using {num_processes} processes for parallel processing")
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_process_image, process_args),
            total=len(process_args),
            desc="Processing detections"
        ))
    
    # Update detections_dict with results
    for image_idx, processed_detections in results:
        detections_dict[image_idx]["detections"] = processed_detections
                
    # write the detections dict to the file
    prInfo(f"Saving detections to {detections_path}")
    write_dic_to_json_file(detections_dict, detections_path)

if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./demo/output/2025_06_13_11_15_31_Theta360", help="Directory to use (default: %(default)s)")
    parser.add_argument("--episodes", "-e", type=str, default="all", help="Episodes to use, separated by commas or 'all' (default: %(default)s)")
    parser.add_argument("--intersection_mask_path", "-imp", type=str, default="./assets/default_mask_intersection_zone.png", help="Intersection mask path (default: %(default)s)")
    parser.add_argument("--min_pixels_intersection", "-mpi", type=int, default=10000, help="Minimum pixels intersection to consider a detection as interacting (default: %(default)s)")
    parser.add_argument("--workers", "-w", type=int, default=0, help="Number of workers to use for parallel processing. Default: 0 (use half of available cores) (default: %(default)s)")
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