
import argparse
import os
import sys

from matplotlib import image

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *
from misc.other_utils import *
from export.export_utils import get_tracks_infos, is_episode_usable, get_contiguous_true_segments, get_image_timestamp, get_image_shape
import pandas as pd
import datetime
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

def process_recording(args, recording_name):
    """ Process a recording
    
    Args:
        recording_name (str): Recording name
    """
    
    if args.splitted_dataset:
        dataset_recording_dir = os.path.join(args.dir, "dataset", recording_name)
        annotations_recording_dir = os.path.join(args.dir, "annotations", recording_name)
    else:
        dataset_recording_dir = os.path.join(args.dir, recording_name)
        annotations_recording_dir = os.path.join(args.dir, recording_name)
    
    episodes_flags_path = os.path.join(dataset_recording_dir, "episodes_flag.json")
    assert os.path.exists(episodes_flags_path), "Episodes flags file does not exist at {}".format(episodes_flags_path)
    episodes_flags = read_json_to_dic(episodes_flags_path)
    
    recording_interaction_zone_center_positions = {}
    
    for episode_name, episode_data in episodes_flags.items():
        valid_flags = is_episode_usable(args, episode_data["flag"])
        if not valid_flags:
            continue

        episode_dir = os.path.join(annotations_recording_dir, "episodes", episode_name)
        assert os.path.isdir(episode_dir), "Episode directory does not exist at {}".format(episode_dir)
        
        if "rosbag2" in recording_name:
            # use the list of file and the default size
            images_list_path = os.path.join(episode_dir, "images_360_filenames.txt")
            assert os.path.exists(images_list_path), "Images list file does not exist at {}".format(images_list_path)
            with open(images_list_path, "r") as f:
                images_list = f.readlines()
            images_list = [image_path.strip() for image_path in images_list]
            image_width = 3840
            image_height = 1920
            interaction_zone_center_x_list = [int(image_width / 2)] * len(images_list)
            interaction_zone_center_y_list = [int(image_height)] * len(images_list)
            recording_interaction_zone_center_positions[episode_name] = {}
            recording_interaction_zone_center_positions[episode_name]["interaction_zone_center_x"] = interaction_zone_center_x_list
            recording_interaction_zone_center_positions[episode_name]["interaction_zone_center_y"] = interaction_zone_center_y_list
        else:
            # use the interaction zone tracking file
            assert os.path.exists(os.path.join(episode_dir, "interaction_zone_tracking_360.json")), "Interaction zone tracking file does not exist at {}".format(os.path.join(episode_dir, "interaction_zone_tracking_360.json"))
            
            interaction_zone_tracking_path = os.path.join(episode_dir, "interaction_zone_tracking_360.json")
            interaction_zone_tracking_dict = read_json_to_dic(interaction_zone_tracking_path)
            interaction_zone_center_x_list = [] # list of length of the number of images in the episode
            interaction_zone_center_y_list = [] # list of length of the number of images in the episode
        
            for image_idx, image_element in enumerate(interaction_zone_tracking_dict):
                interaction_zone_xmin = image_element["detections"][0]["xmin"]
                interaction_zone_ymin = image_element["detections"][0]["ymin"]
                interaction_zone_xmax = image_element["detections"][0]["xmax"]
                interaction_zone_ymax = image_element["detections"][0]["ymax"]
                image_width = image_element["image_shape"][1]
                image_height = image_element["image_shape"][0]
                interaction_zone_center_x = int((interaction_zone_xmin + interaction_zone_xmax) / 2)
                interaction_zone_center_y = int((interaction_zone_ymin + interaction_zone_ymax) / 2)
                
                if interaction_zone_center_x >= 2*image_width or interaction_zone_center_x < -2*image_width:
                    raise ValueError(f"Interaction zone center x is out of bounds for image {image_element['image_file']}")
                
                if interaction_zone_center_x < 0:
                    interaction_zone_center_x = interaction_zone_center_x + image_width
                
                if interaction_zone_center_x >= image_width:
                    interaction_zone_center_x = interaction_zone_center_x - image_width
                
                interaction_zone_center_x_list.append(interaction_zone_center_x)
                interaction_zone_center_y_list.append(interaction_zone_center_y)
                
            recording_interaction_zone_center_positions[episode_name] = {
                "interaction_zone_center_x": interaction_zone_center_x_list,
                "interaction_zone_center_y": interaction_zone_center_y_list
            }
    
    return recording_interaction_zone_center_positions
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./output/")
    parser.add_argument("--input", "-i", type=str, default="images_360", help="Input type (only supports images_360)")
    parser.add_argument("--recordings", "-r", type=str, default="all", help="Recordings to export, all or a list of recordings names")
    parser.add_argument("--splitted_dataset", "-sd", action="store_true", default=False, help="Use splitted dataset mode, the data and the annotations are separated into different directories (default: %(default)s)")
    args = parser.parse_args()

    assert args.input in ["images_360"], "Invalid input type, expected images_360"
    assert os.path.isdir(args.dir), "Directory should exist at {}".format(args.dir)
    
    if args.recordings == "all":
        # only keep directories
        if args.splitted_dataset:
            recordings_list = [f for f in os.listdir(os.path.join(args.dir, "dataset")) if os.path.isdir(os.path.join(args.dir, "dataset", f))]
        else:
            recordings_list = [f for f in os.listdir(os.path.join(args.dir)) if os.path.isdir(os.path.join(args.dir, f))]
    else:
        recordings_list = args.recordings.split(",")
        
    recordings_list.sort()
    recordings_list = [r for r in recordings_list if not r.startswith(".")]
    
    prInfo(f"Recordings list: {recordings_list}")
    
    all_recordings_interaction_zone_center_positions = {}
    
    for recording in tqdm(recordings_list, desc="Processing recordings", total=len(recordings_list)):
        recording_interaction_zone_center_positions = process_recording(args, recording)
        all_recordings_interaction_zone_center_positions[recording] = recording_interaction_zone_center_positions

    write_dic_to_json_file(all_recordings_interaction_zone_center_positions, os.path.join(args.dir, "interaction_zone_center_positions.json"))
    
    # # validity check first - parallelized
    # prInfo(f"Checking {len(recordings_list)} recordings validity (using {args.workers} workers)")
    # recording_args_list = []
    # for recording in recordings_list:
    #     if args.splitted_dataset:
    #         assert os.path.isdir(os.path.join(args.dir, "dataset", recording)), "Dataset directory {} should exist and be a directory".format(os.path.join(args.dir, "dataset", recording))
    #         dataset_recording_dir = os.path.join(args.dir, "dataset", recording)
    #         assert os.path.isdir(os.path.join(args.dir, "annotations", recording)), "Annotations directory {} should exist and be a directory".format(os.path.join(args.dir, "annotations", recording))
    #         annotations_recording_dir = os.path.join(args.dir, "annotations", recording)
    #     else:
    #         dataset_recording_dir = os.path.join(args.dir, recording)
    #         assert os.path.isdir(dataset_recording_dir), "Recording {} should exist and be a directory".format(dataset_recording_dir)
    #         annotations_recording_dir = None
    #     recording_args_list.append((args, dataset_recording_dir, annotations_recording_dir))
    
    # # Check validity in parallel
    # with mp.Pool(processes=args.workers) as pool:
    #     validity_results = pool.map(check_recording_validity_wrapper, recording_args_list)
    
    # # Only extract tracks if there is at least one valid episode
    # valid_recordings = []
    # valid_recording_args_list = []
    # for recording, n_valid_episodes, (args, dataset_recording_dir, annotations_recording_dir) in zip(recordings_list, validity_results, recording_args_list):
    #     if n_valid_episodes > 0:
    #         prSuccess(f"Recording {recording} is valid ({n_valid_episodes} episodes are valid) and will be exported later")
    #         valid_recordings.append(recording)
    #         valid_recording_args_list.append((args, dataset_recording_dir, annotations_recording_dir))
    #     else:
    #         prWarning(f"Recording {recording} has no valid episodes and will be skipped")
    
    # if len(valid_recordings) == 0:
    #     prWarning("No valid recordings found, exiting...")
    #     exit(1)
    
    # # Create output directory before processing
    # timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # dataset_name = f"exported_dataset_{args.input}_{timestamp}"
    # dirname = args.dir.split("/")[-1] if args.dir.split("/")[-1] != "" else args.dir.split("/")[-2]
    # dataset_dir = os.path.join(args.dir, "..", dirname+"_exported_datasets", dataset_name)
    # if not os.path.exists(os.path.join(args.dir, "..", dirname+"_exported_datasets")):
    #     prInfo("Creating exported datasets directory at absolute path {}".format(os.path.join(args.dir, "..", dirname+"_exported_datasets")))
    #     os.makedirs(os.path.join(args.dir, "..", dirname+"_exported_datasets"))
    # os.makedirs(dataset_dir, exist_ok=True)
    
    # # Process recordings sequentially, parallelize at episode level, save immediately to free RAM
    # prInfo(f"Processing {len(valid_recordings)} recordings sequentially (parallelizing at episode level with {args.workers} workers)")
    
    # total_recordings = len(valid_recording_args_list)
    # saved_count = 0
    
    # for count, (recording, (args_item, dataset_recording_dir, annotations_recording_dir)) in enumerate(zip(valid_recordings, valid_recording_args_list)):
    #     prInfo(f"[{count+1}/{total_recordings}] Processing recording {recording}...")
        
    #     # Get recording data with episode-level parallelization
    #     recording_df = get_recording_data(args_item, dataset_recording_dir, annotations_recording_dir, workers=args.workers)
        
    #     if len(recording_df) > 0:
    #         prSuccess(f"Got data for recording {recording} (entries : {len(recording_df)}, unique tracks : {len(recording_df['unique_track_identifier'].unique())})")
            
    #         # Save immediately to free RAM
    #         save_recording_name = f"data-{recording}-{count:04d}-of-{total_recordings:04d}.csv"
    #         recording_df.to_csv(os.path.join(dataset_dir, save_recording_name), index=False)
    #         prSuccess(f"Saved recording {recording} to {save_recording_name}")
    #         saved_count += 1
            
    #         # Free memory
    #         del recording_df
    #     else:
    #         prWarning(f"Recording {recording} has no valid data, skipping save")
    
    # if saved_count == 0:
    #     prWarning("No recordings with valid data were saved")
    # else:
    #     prSuccess(f"Exported {saved_count} recordings to {dataset_dir}")