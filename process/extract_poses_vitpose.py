import torch
import numpy as np
import argparse
import os
import sys
import datetime
import torch.multiprocessing as mp
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
        
        detections = detections_dict[image_idx]["detections"]
        
        person_boxes = []
        valid_boxes_idxs = []
        invalid_boxes_idxs = []
        for idx, det in enumerate(detections):
            box = [det["xmin"], det["ymin"], det["xmax"], det["ymax"]]
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]] # convert to COCO format
            
            if box[2] > 50.0 and box[3] > 100.0:
                # box big enough
                person_boxes.append(box)
                valid_boxes_idxs.append(idx)
            else:
                invalid_boxes_idxs.append(idx)

        if len(valid_boxes_idxs) > 0:
            
            person_boxes = np.array(person_boxes) # NDet, 4
            inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device) # NOTE : this does not allow to put the masks on the images / Must be sure that the box contains only one person

            with torch.no_grad():
                outputs = model(**inputs)

            pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
            del inputs, outputs
            torch.cuda.empty_cache()
            image_pose_result = pose_results[0]  # results for first image : list of dict, [{"keypoints": [NKP, 2], "scores": [NKP]}, ...]
            
            # print(image_pose_result)
            
            for pose, detection_idx in zip(image_pose_result, valid_boxes_idxs):
                keypoints = pose["keypoints"].cpu().numpy().flatten().tolist()
                keypoints = [int(coord) for coord in keypoints]
                scores = pose["scores"].cpu().numpy().flatten().tolist()
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
    