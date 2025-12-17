
# TODO : 
# For SSUP files
# Add the image size
# Rename the detections files to : annotations_360.json

# For the classif Shelfy ones
# Remove the face keys
# Rename the engagement level
# Add the intersection pixels
# Rename the detections files to : annotations_360.json

# 2023_07_12_albee_square_landfill_1 removed (not 360)

## SSUP processing

SSUP_DIRECTORY_EPISODES_JSON = "/media/raphael/hucedisk3/SSUP/FullVideos/extracted/"

OUTPUT_DIRECTORY = "/media/raphael/hucedisk3/HUI360-Videos/"

import os
import sys
import shutil
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *

recordings_list_episodes_json = os.listdir(SSUP_DIRECTORY_EPISODES_JSON)
recordings_list_episodes_json.sort()

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "recordings"))

for recording_name in recordings_list_episodes_json:
    # first get the episodes and flags metadata
    episodes_json_path = os.path.join(SSUP_DIRECTORY_EPISODES_JSON, recording_name, "episodes.json")
    episodes_flags_json_path = os.path.join(SSUP_DIRECTORY_EPISODES_JSON, recording_name, "episodes_flag.json")
    
    assert(os.path.exists(episodes_json_path)), f"Episodes json file does not exist: {episodes_json_path}"
    assert(os.path.exists(episodes_flags_json_path)), f"Episodes flags json file does not exist: {episodes_flags_json_path}"
    
    episodes_json_dict = read_json_to_dic(episodes_json_path)
    episodes_flags_json_dict = read_json_to_dic(episodes_flags_json_path)
    assert(len(episodes_json_dict) == len(episodes_flags_json_dict)), f"Number of episodes in the episodes json and episodes flags json must be the same"
    
    recording_output_dir = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name)
    if not os.path.exists(recording_output_dir):
        os.makedirs(recording_output_dir)
    assert(len(episodes_json_dict) == len(episodes_flags_json_dict)), f"Number of episodes in the episodes json and episodes flags json must be the same"
    
    shutil.copy2(episodes_json_path, os.path.join(recording_output_dir, "episodes.json"))
    shutil.copy2(episodes_flags_json_path, os.path.join(recording_output_dir, "episodes_flag.json"))
    
    prSuccess(f"Copied episodes json and episodes flags json to {recording_output_dir}")
    
    
    for episode_name, flags_dict in episodes_flags_json_dict.items():
        
        interaction_zone_tracking_path = os.path.join(SSUP_DIRECTORY_EPISODES_JSON, recording_name, "episodes", episode_name, "interact_zone_tracking_360.json")
        detections_path = os.path.join(SSUP_DIRECTORY_EPISODES_JSON, recording_name, "episodes", episode_name, "sam2_detections_360.json")
        assert(os.path.exists(interaction_zone_tracking_path)), f"Interaction zone tracking file does not exist: {interaction_zone_tracking_path}"
        assert(os.path.exists(detections_path)), f"Detections file does not exist: {detections_path}"

    
        tic = time.time()    
        interaction_zone_tracking_dict = read_json_to_dic(interaction_zone_tracking_path)
        prInfo(f"Got {len(interaction_zone_tracking_dict)} interaction zone tracking for {recording_name}/{episode_name}")
        for image_idx in range(len(interaction_zone_tracking_dict)):
            interaction_zone_tracking_dict[image_idx]["image_shape"] = [1920, 3840, 3]
        output_interaction_zone_tracking_dir = os.path.join(recording_output_dir, "episodes", episode_name)
        if not os.path.exists(output_interaction_zone_tracking_dir):
            os.makedirs(output_interaction_zone_tracking_dir)
        output_interaction_zone_tracking_name ="interaction_zone_tracking_360.json"
        output_interaction_zone_tracking_path = os.path.join(output_interaction_zone_tracking_dir, output_interaction_zone_tracking_name)
        write_dic_to_json_file(interaction_zone_tracking_dict, output_interaction_zone_tracking_path)
        tac = time.time()
        prSuccess(f"Saved interaction zone tracking to {output_interaction_zone_tracking_path} (total time : {tac-tic:.2f} seconds)")
    
    
        tic = time.time()
        detections_dict = read_json_to_dic(detections_path)
        prInfo(f"Got {len(detections_dict)} image idxs for {recording_name}/{episode_name}")
        for image_idx in range(len(detections_dict)):
            detections_dict[image_idx]["image_shape"] = [1920, 3840, 3]
        output_detections_dir = os.path.join(recording_output_dir, "episodes", episode_name)
        if not os.path.exists(output_detections_dir):
            os.makedirs(output_detections_dir)
        output_detections_name ="annotations_360.json"
        output_detections_path = os.path.join(output_detections_dir, output_detections_name)
        write_dic_to_json_file(detections_dict, output_detections_path)
        tac = time.time()
        prSuccess(f"Saved detections to {output_detections_path} (total time : {tac-tic:.2f} seconds)")
                
        # for idx in range(len(detections_dict)):
        #     image_file = detections_dict[idx]["image_file"]
        #     image_file_with_sapiens = detections_dict_with_sapiens[idx]["image_file"]
        #     assert(image_file == image_file_with_sapiens), f"Image file in the detections and detections with sapiens json must be the same, different at index {idx} (image file : {image_file})"
            
        #     detections_in_detection_dict = detections_dict[idx]["detections"]
        #     detections_in_detection_dict_with_sapiens = detections_dict_with_sapiens[idx]["detections"]    
            
        #     if len(detections_in_detection_dict) != len(detections_in_detection_dict_with_sapiens):
        #         prError(f"Number of detections in the detections and detections with sapiens json must be the same, different at index {idx} (image file : {detections_dict[idx]['image_file']})")
                
        #         print([det["track_id"] for det in detections_in_detection_dict])
        #         print([det["track_id"] for det in detections_in_detection_dict_with_sapiens])
                
        #         raise ValueError(f"Number of detections in the detections and detections with sapiens json must be the same, different at index {idx} (image file : {detections_dict[idx]['image_file']})")
        #     else:
        #         pass
            
        #  assert(len(detections_in_detection_dict) == len(detections_in_detection_dict_with_sapiens)), f"Number of detections in the detections and detections with sapiens json must be the same, different at index {idx} (image file : {detections_dict[idx]['image_file']})"
        
            