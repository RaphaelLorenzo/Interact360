
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

## Shelfy processing

SHELFY_DIRECTORY_EPISODES_JSON = "/media/raphael/hucedisk/ShelfyInteractFull/data_extractor/"
OUTPUT_DIRECTORY = "/media/raphael/hucedisk3/HUI360-Videos/"

import os
import sys
import shutil
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *

recordings_list_episodes_json = os.listdir(SHELFY_DIRECTORY_EPISODES_JSON)
recordings_list_episodes_json.sort()

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "recordings"))


UNEXPORTED_RECORDINGS = ["rosbag2_2025_10_07-16_00_15"]

for recording_name in recordings_list_episodes_json:
    
    if recording_name in UNEXPORTED_RECORDINGS:
        prWarning(f"Skipping unexported recording: {recording_name}")
        continue
    
    # first get the episodes and flags metadata
    episodes_json_path = os.path.join(SHELFY_DIRECTORY_EPISODES_JSON, recording_name, "episodes.json")
    episodes_flags_json_path = os.path.join(SHELFY_DIRECTORY_EPISODES_JSON, recording_name, "episodes_flag.json")
    
    assert(os.path.exists(episodes_json_path)), f"Episodes json file does not exist: {episodes_json_path}"
    assert(os.path.exists(episodes_flags_json_path)), f"Episodes flags json file does not exist: {episodes_flags_json_path}"
    
    episodes_json_dict = read_json_to_dic(episodes_json_path)
    episodes_flags_json_dict = read_json_to_dic(episodes_flags_json_path)
    assert(len(episodes_json_dict) == len(episodes_flags_json_dict)), f"Number of episodes in the episodes json and episodes flags json must be the same"
    
    recording_output_dir = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name)
    if not os.path.exists(recording_output_dir):
        os.makedirs(recording_output_dir)
    
    # update the episodes json to add the session and episode names
    for epidx, ep in enumerate(episodes_json_dict):
        episodes_json_dict[epidx]["session"] = recording_name
        episodes_json_dict[epidx]["episode"] = f"{epidx:04d}"
    write_dic_to_json_file(episodes_json_dict, os.path.join(recording_output_dir, "episodes.json"))
    
    # just copy the episodes flags json already good format
    shutil.copy2(episodes_flags_json_path, os.path.join(recording_output_dir, "episodes_flag.json"))
    
    prSuccess(f"Copied episodes json and episodes flags json to {recording_output_dir}")
    
    
    for episode_name, flags_dict in episodes_flags_json_dict.items():
        
        detections_path = os.path.join(SHELFY_DIRECTORY_EPISODES_JSON, recording_name, "episodes", episode_name, "detections_sam2_360.json")
        assert(os.path.exists(detections_path)), f"Detections file does not exist: {detections_path}"
            
        tic = time.time()
        detections_dict = read_json_to_dic(detections_path)
        prInfo(f"Got {len(detections_dict)} image idxs for {recording_name}/{episode_name}")
        total_eng_replaced = 0
        total_eng_added = 0
        total_face_popped = 0
        total_head_rot_mat_popped = 0
        total_mask_file_popped = 0
        total_track_type_popped = 0
        for image_idx in range(len(detections_dict)):
            
            for det_idx in range(len(detections_dict[image_idx]["detections"])):
                current_det = detections_dict[image_idx]["detections"][det_idx]
            
                if "engagement_level_auto_360" in current_det:
                    # rename engagement_level_auto_360 to engagement_level
                    detections_dict[image_idx]["detections"][det_idx]["engagement_level"] = detections_dict[image_idx]["detections"][det_idx].pop("engagement_level_auto_360")
                    total_eng_replaced += 1
                else:
                    detections_dict[image_idx]["detections"][det_idx]["engagement_level"] = 0
                    total_eng_added += 1
            
                if "face_confidence" in current_det:
                    detections_dict[image_idx]["detections"][det_idx].pop("face_confidence")
                    detections_dict[image_idx]["detections"][det_idx].pop("face_xmin")
                    detections_dict[image_idx]["detections"][det_idx].pop("face_ymin")
                    detections_dict[image_idx]["detections"][det_idx].pop("face_xmax")
                    detections_dict[image_idx]["detections"][det_idx].pop("face_ymax")
                    total_face_popped += 1
                    
                if "head_rot_mat" in current_det:
                    detections_dict[image_idx]["detections"][det_idx].pop("head_rot_mat")
                    total_head_rot_mat_popped += 1
                    
                if "mask_file" in current_det:
                    detections_dict[image_idx]["detections"][det_idx].pop("mask_file")
                    total_mask_file_popped += 1
                    
                if "track_type" in current_det:
                    detections_dict[image_idx]["detections"][det_idx].pop("track_type")
                    total_track_type_popped += 1
                    
                detections_dict[image_idx]["detections"][det_idx]["intersection_pixels"] = -1
            
        prInfo(f"Engagement : replaced {total_eng_replaced} / added {total_eng_added} engagement levels | Popped {total_face_popped} faces, {total_head_rot_mat_popped} head rot mat, {total_mask_file_popped} mask file keys, {total_track_type_popped} track type keys")
        
        output_detections_dir = os.path.join(recording_output_dir, "episodes", episode_name)
        if not os.path.exists(output_detections_dir):
            os.makedirs(output_detections_dir)
        output_detections_name ="annotations_360.json"
        output_detections_path = os.path.join(output_detections_dir, output_detections_name)
        write_dic_to_json_file(detections_dict, output_detections_path)
        tac = time.time()
        prSuccess(f"Saved detections to {output_detections_path} (total time : {tac-tic:.2f} seconds)")
                
