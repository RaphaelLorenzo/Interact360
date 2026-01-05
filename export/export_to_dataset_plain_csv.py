# This script aims at exporting in HUI format

EXPECTED_DET_KEYS = ["track_id", 
                     "validity", 
                     "mask_rle", 
                     "mask_size",
                     "xmin", 
                     "ymin",
                     "xmax",
                     "ymax",
                     "keypoints_vitpose",
                     "scores_vitpose",
                     "engagement_level",
                    #  "keypoints_sapiens_308_06b",
                    #  "scores_sapiens_308_06b",
                    #  "face_xmin",
                    #  "face_ymin",
                    #  "face_xmax",
                    #  "face_ymax",
                    #  "face_confidence",
                    #  "head_rot_mat"
                     ]

VITPOSE_KEYPOINTS_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
VITPOSE_KEYPOINTS_NAMES = ["vitpose_" + kpname for kpname in VITPOSE_KEYPOINTS_NAMES]
assert(len(VITPOSE_KEYPOINTS_NAMES) == 17)
SAPIENS_KEYPOINTS_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel", "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint", "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint", "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint", "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint", "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint", "right_wrist", "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint", "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint", "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint", "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint", "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint", "left_wrist", "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa", "left_acromion", "right_acromion", "neck", "center_of_glabella", "center_of_nose_root", "tip_of_nose_bridge", "midpoint_1_of_nose_bridge", "midpoint_2_of_nose_bridge", "midpoint_3_of_nose_bridge", "center_of_labiomental_groove", "tip_of_chin", "upper_startpoint_of_r_eyebrow", "lower_startpoint_of_r_eyebrow", "end_of_r_eyebrow", "upper_midpoint_1_of_r_eyebrow", "lower_midpoint_1_of_r_eyebrow", "upper_midpoint_2_of_r_eyebrow", "upper_midpoint_3_of_r_eyebrow", "lower_midpoint_2_of_r_eyebrow", "lower_midpoint_3_of_r_eyebrow", "upper_startpoint_of_l_eyebrow", "lower_startpoint_of_l_eyebrow", "end_of_l_eyebrow", "upper_midpoint_1_of_l_eyebrow", "lower_midpoint_1_of_l_eyebrow", "upper_midpoint_2_of_l_eyebrow", "upper_midpoint_3_of_l_eyebrow", "lower_midpoint_2_of_l_eyebrow", "lower_midpoint_3_of_l_eyebrow", "l_inner_end_of_upper_lash_line", "l_outer_end_of_upper_lash_line", "l_centerpoint_of_upper_lash_line", "l_midpoint_2_of_upper_lash_line", "l_midpoint_1_of_upper_lash_line", "l_midpoint_6_of_upper_lash_line", "l_midpoint_5_of_upper_lash_line", "l_midpoint_4_of_upper_lash_line", "l_midpoint_3_of_upper_lash_line", "l_outer_end_of_upper_eyelid_line", "l_midpoint_6_of_upper_eyelid_line", "l_midpoint_2_of_upper_eyelid_line", "l_midpoint_5_of_upper_eyelid_line", "l_centerpoint_of_upper_eyelid_line", "l_midpoint_4_of_upper_eyelid_line", "l_midpoint_1_of_upper_eyelid_line", "l_midpoint_3_of_upper_eyelid_line", "l_midpoint_6_of_upper_crease_line", "l_midpoint_2_of_upper_crease_line", "l_midpoint_5_of_upper_crease_line", "l_centerpoint_of_upper_crease_line", "l_midpoint_4_of_upper_crease_line", "l_midpoint_1_of_upper_crease_line", "l_midpoint_3_of_upper_crease_line", "r_inner_end_of_upper_lash_line", "r_outer_end_of_upper_lash_line", "r_centerpoint_of_upper_lash_line", "r_midpoint_1_of_upper_lash_line", "r_midpoint_2_of_upper_lash_line", "r_midpoint_3_of_upper_lash_line", "r_midpoint_4_of_upper_lash_line", "r_midpoint_5_of_upper_lash_line", "r_midpoint_6_of_upper_lash_line", "r_outer_end_of_upper_eyelid_line", "r_midpoint_3_of_upper_eyelid_line", "r_midpoint_1_of_upper_eyelid_line", "r_midpoint_4_of_upper_eyelid_line", "r_centerpoint_of_upper_eyelid_line", "r_midpoint_5_of_upper_eyelid_line", "r_midpoint_2_of_upper_eyelid_line", "r_midpoint_6_of_upper_eyelid_line", "r_midpoint_3_of_upper_crease_line", "r_midpoint_1_of_upper_crease_line", "r_midpoint_4_of_upper_crease_line", "r_centerpoint_of_upper_crease_line", "r_midpoint_5_of_upper_crease_line", "r_midpoint_2_of_upper_crease_line", "r_midpoint_6_of_upper_crease_line", "l_inner_end_of_lower_lash_line", "l_outer_end_of_lower_lash_line", "l_centerpoint_of_lower_lash_line", "l_midpoint_2_of_lower_lash_line", "l_midpoint_1_of_lower_lash_line", "l_midpoint_6_of_lower_lash_line", "l_midpoint_5_of_lower_lash_line", "l_midpoint_4_of_lower_lash_line", "l_midpoint_3_of_lower_lash_line", "l_outer_end_of_lower_eyelid_line", "l_midpoint_6_of_lower_eyelid_line", "l_midpoint_2_of_lower_eyelid_line", "l_midpoint_5_of_lower_eyelid_line", "l_centerpoint_of_lower_eyelid_line", "l_midpoint_4_of_lower_eyelid_line", "l_midpoint_1_of_lower_eyelid_line", "l_midpoint_3_of_lower_eyelid_line", "r_inner_end_of_lower_lash_line", "r_outer_end_of_lower_lash_line", "r_centerpoint_of_lower_lash_line", "r_midpoint_1_of_lower_lash_line", "r_midpoint_2_of_lower_lash_line", "r_midpoint_3_of_lower_lash_line", "r_midpoint_4_of_lower_lash_line", "r_midpoint_5_of_lower_lash_line", "r_midpoint_6_of_lower_lash_line", "r_outer_end_of_lower_eyelid_line", "r_midpoint_3_of_lower_eyelid_line", "r_midpoint_1_of_lower_eyelid_line", "r_midpoint_4_of_lower_eyelid_line", "r_centerpoint_of_lower_eyelid_line", "r_midpoint_5_of_lower_eyelid_line", "r_midpoint_2_of_lower_eyelid_line", "r_midpoint_6_of_lower_eyelid_line", "tip_of_nose", "bottom_center_of_nose", "r_outer_corner_of_nose", "l_outer_corner_of_nose", "inner_corner_of_r_nostril", "outer_corner_of_r_nostril", "upper_corner_of_r_nostril", "inner_corner_of_l_nostril", "outer_corner_of_l_nostril", "upper_corner_of_l_nostril", "r_outer_corner_of_mouth", "l_outer_corner_of_mouth", "center_of_cupid_bow", "center_of_lower_outer_lip", "midpoint_1_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip", "midpoint_1_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip", "midpoint_3_of_upper_outer_lip", "midpoint_4_of_upper_outer_lip", "midpoint_5_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip", "midpoint_3_of_lower_outer_lip", "midpoint_4_of_lower_outer_lip", "midpoint_5_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip", "r_inner_corner_of_mouth", "l_inner_corner_of_mouth", "center_of_upper_inner_lip", "center_of_lower_inner_lip", "midpoint_1_of_upper_inner_lip", "midpoint_2_of_upper_inner_lip", "midpoint_1_of_lower_inner_lip", "midpoint_2_of_lower_inner_lip", "midpoint_3_of_upper_inner_lip", "midpoint_4_of_upper_inner_lip", "midpoint_5_of_upper_inner_lip", "midpoint_6_of_upper_inner_lip", "midpoint_3_of_lower_inner_lip", "midpoint_4_of_lower_inner_lip", "midpoint_5_of_lower_inner_lip", "midpoint_6_of_lower_inner_lip", "l_top_end_of_inferior_crus", "l_top_end_of_superior_crus", "l_start_of_antihelix", "l_end_of_antihelix", "l_midpoint_1_of_antihelix", "l_midpoint_1_of_inferior_crus", "l_midpoint_2_of_antihelix", "l_midpoint_3_of_antihelix", "l_point_1_of_inner_helix", "l_point_2_of_inner_helix", "l_point_3_of_inner_helix", "l_point_4_of_inner_helix", "l_point_5_of_inner_helix", "l_point_6_of_inner_helix", "l_point_7_of_inner_helix", "l_highest_point_of_antitragus", "l_bottom_point_of_tragus", "l_protruding_point_of_tragus", "l_top_point_of_tragus", "l_start_point_of_crus_of_helix", "l_deepest_point_of_concha", "l_tip_of_ear_lobe", "l_midpoint_between_22_15", "l_bottom_connecting_point_of_ear_lobe", "l_top_connecting_point_of_helix", "l_point_8_of_inner_helix", "r_top_end_of_inferior_crus", "r_top_end_of_superior_crus", "r_start_of_antihelix", "r_end_of_antihelix", "r_midpoint_1_of_antihelix", "r_midpoint_1_of_inferior_crus", "r_midpoint_2_of_antihelix", "r_midpoint_3_of_antihelix", "r_point_1_of_inner_helix", "r_point_8_of_inner_helix", "r_point_3_of_inner_helix", "r_point_4_of_inner_helix", "r_point_5_of_inner_helix", "r_point_6_of_inner_helix", "r_point_7_of_inner_helix", "r_highest_point_of_antitragus", "r_bottom_point_of_tragus", "r_protruding_point_of_tragus", "r_top_point_of_tragus", "r_start_point_of_crus_of_helix", "r_deepest_point_of_concha", "r_tip_of_ear_lobe", "r_midpoint_between_22_15", "r_bottom_connecting_point_of_ear_lobe", "r_top_connecting_point_of_helix", "r_point_2_of_inner_helix", "l_center_of_iris", "l_border_of_iris_3", "l_border_of_iris_midpoint_1", "l_border_of_iris_12", "l_border_of_iris_midpoint_4", "l_border_of_iris_9", "l_border_of_iris_midpoint_3", "l_border_of_iris_6", "l_border_of_iris_midpoint_2", "r_center_of_iris", "r_border_of_iris_3", "r_border_of_iris_midpoint_1", "r_border_of_iris_12", "r_border_of_iris_midpoint_4", "r_border_of_iris_9", "r_border_of_iris_midpoint_3", "r_border_of_iris_6", "r_border_of_iris_midpoint_2", "l_center_of_pupil", "l_border_of_pupil_3", "l_border_of_pupil_midpoint_1", "l_border_of_pupil_12", "l_border_of_pupil_midpoint_4", "l_border_of_pupil_9", "l_border_of_pupil_midpoint_3", "l_border_of_pupil_6", "l_border_of_pupil_midpoint_2", "r_center_of_pupil", "r_border_of_pupil_3", "r_border_of_pupil_midpoint_1", "r_border_of_pupil_12", "r_border_of_pupil_midpoint_4", "r_border_of_pupil_9", "r_border_of_pupil_midpoint_3", "r_border_of_pupil_6", "r_border_of_pupil_midpoint_2"]
SAPIENS_KEYPOINTS_NAMES = ["sapiens_308_" + kpname for kpname in SAPIENS_KEYPOINTS_NAMES]
assert(len(SAPIENS_KEYPOINTS_NAMES) == 308)

import argparse
import os
import sys

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *
from misc.other_utils import *
from export.export_utils import get_tracks_infos, is_episode_usable, get_contiguous_true_segments, get_image_timestamp, get_image_shape
import pandas as pd
import datetime
import multiprocessing as mp
import numpy as np

def check_recording_validity(args, dataset_recording_dir, annotations_recording_dir=None):
    """ Check that the recording is valid by checking the episodes flags and the detections (make sure they have the necessary keys)

    Args:
        args (argparse.Namespace): Arguments
        dataset_recording_dir (str): Path to the dataset recording directory (contains episodes_flag.json)
        annotations_recording_dir (str): Path to the annotations recording directory (contains episodes with detections), defaults to dataset_recording_dir if None

    Returns:
        int: Number of valid episodes
    """
    
    if annotations_recording_dir is None:
        annotations_recording_dir = dataset_recording_dir
    
    # Read the episodes flags to check that if they should be discarded altogether
    episodes_flags_path = os.path.join(dataset_recording_dir, "episodes_flag.json")
    assert os.path.exists(episodes_flags_path), "Episodes flags file does not exist at {}".format(episodes_flags_path)
    episodes_flags = read_json_to_dic(episodes_flags_path)
    
    expected_det_keys = EXPECTED_DET_KEYS.copy()
    
    n_valid_episodes = 0
    for episode_name, episode_data in episodes_flags.items():
        valid_flags = is_episode_usable(args, episode_data["flag"])
        if not valid_flags:
            # ignore the full episode
            continue
        else:
            n_valid_episodes += 1
    
        # check that we have the necessary inputs in the episode
        episode_dir = os.path.join(annotations_recording_dir, "episodes", episode_name)
        assert os.path.isdir(episode_dir), "Episode directory does not exist at {}".format(episode_dir)
        assert os.path.exists(os.path.join(episode_dir, "annotations_360.json")), "Detections file does not exist at {}".format(os.path.join(episode_dir, "annotations_360.json"))
        
        # check that the images directory exists except if we allow empty dirs to work only with labels
        if not args.allow_empty_dirs:
            assert os.path.exists(os.path.join(episode_dir, "images_360")), "Images 360 directory does not exist at {}".format(os.path.join(episode_dir, "images_360"))
            
        # read the detections
        detections_path = os.path.join(episode_dir, "annotations_360.json")
        detections_dict = read_json_to_dic(detections_path)
        
        # eventually check that the images exist and that the number of detections and images is the same except if we allow empty dirs to work only with labels
        images_dir = os.path.join(episode_dir, "images_360")
        if not args.allow_empty_dirs:
            images_files = os.listdir(images_dir)
            images_files = [os.path.join(images_dir, f) for f in images_files if f.endswith(".jpg")]
            images_files.sort()
            assert len(detections_dict) == len(images_files), "Number of detections and images should be the same ({} vs {})".format(len(detections_dict), len(images_files))
        else:   
            images_files = []
            
        # check that the detections are valid
        for image_idx, image_element in enumerate(detections_dict):
            for det_idx, det in enumerate(image_element["detections"]):
                assert("validity" in det.keys()), "Validity not found in detection (got : {})".format(det.keys())
                validity = det["validity"]
                if validity != "valid":
                    # ignore this detection since it is not valid anyway, we will not use it later on
                    continue
                for key in expected_det_keys:
                    track_id = det["track_id"]
                    image_file = image_element["image_file"]
                    assert key in det.keys(), "[{}][{}][{}][{} - track_id : {}] Key {} not found in detection (got : {}, validity : {}, image_file : {})".format(os.path.basename(dataset_recording_dir), episode_name, image_idx, det_idx, track_id, key, det.keys(), validity, image_file)
        
        # print(os.path.basename(dataset_recording_dir), episode_name, "valid")
    
    if n_valid_episodes == 0:
        prWarning("No valid episodes found in recording {}, will not export it".format(os.path.basename(dataset_recording_dir)))
    else:
        prSuccess(f"{os.path.basename(dataset_recording_dir)} : {n_valid_episodes}/{len(episodes_flags)} valid episodes")
    
    return n_valid_episodes

def get_track_data(detections_dict, track_info, track_first_interaction_index, valid_segments):
    """ Get the data of a track as pd.DataFrame with each line corresponding to one timestep and the columns being features (keypoints, box coordinates, scores, etc.)

    Args:
        detections_dict (list): _description_
        track_info (_type_): _description_
        track_first_interaction_index (_type_): _description_
        valid_segments (list): list of contiguous valid segments of the form [(start,length), (start,length), ...] if the track with start and length indexed relatively to the beginning of the track in image indexes

    Returns:
        pd.DataFrame: DataFrame with each line corresponding to one timestep and the columns being features (keypoints, box coordinates, scores, etc.)
    """

    if track_first_interaction_index == -1:
        first_interaction_timestamp = None
    else:
        first_interaction_associated_image_file = detections_dict[track_info["images_indexes"][track_first_interaction_index]]["image_file"]
        first_interaction_timestamp = get_image_timestamp(first_interaction_associated_image_file)
    
    track_data = {"image_file":[], 
                  "image_index":[], 
                  "validity":[], 
                  "current_segment":[], 
                  "total_segments":[], 
                  "position_in_segment":[], 
                  "length_of_current_segment": [], 
                  "timestamp":[], 
                  "timestamp_sec":[], 
                  "timestamp_track": [], 
                  "engagement":[], 
                  "time_to_first_interaction":[], 
                  "mask_rle":[], 
                  "mask_size":[], 
                  "xmin":[], 
                  "ymin":[], 
                  "xmax":[], 
                  "ymax":[]}
    
    for vitpose_keypoint_name in VITPOSE_KEYPOINTS_NAMES:
        track_data[vitpose_keypoint_name+"_x"] = []
        track_data[vitpose_keypoint_name+"_y"] = []
        track_data[vitpose_keypoint_name+"_score"] = []
    
    for sapiens_keypoint_name in SAPIENS_KEYPOINTS_NAMES:
        track_data[sapiens_keypoint_name+"_x"] = []
        track_data[sapiens_keypoint_name+"_y"] = []
        track_data[sapiens_keypoint_name+"_score"] = []
        
    images_indexes = track_info["images_indexes"]
    first_image_index = min(images_indexes)
    det_indexes = track_info["det_indexes"]
    track_first_image_timestamp = None
    for idx_in_track, image_idx, det_idx in zip(range(len(images_indexes)), images_indexes, det_indexes):
        
        # get the current segment
        current_segment_idx = -1
        position_in_segment = -1
        length_of_current_segment = -1
        relative_image_idx = image_idx - first_image_index
        for segment_idx, segment in enumerate(valid_segments):
            if relative_image_idx >= segment[0] and relative_image_idx < segment[0] + segment[1]:
                current_segment_idx = segment_idx
                position_in_segment = relative_image_idx - segment[0]
                length_of_current_segment = segment[1]
                break
        
        image_file = detections_dict[image_idx]["image_file"]
        timestamp = get_image_timestamp(image_file)
        if track_first_image_timestamp is None:
            track_first_image_timestamp = timestamp
            
        time_since_track_first_image = timestamp - track_first_image_timestamp
        time_since_track_first_image_sec = time_since_track_first_image.total_seconds()
        
        if first_interaction_timestamp is not None:
            time_to_first_interaction = first_interaction_timestamp - timestamp
            time_to_first_interaction_sec = time_to_first_interaction.total_seconds()
        else:
            time_to_first_interaction_sec = 99999.0
            
        timestamp_sec = timestamp.timestamp()
        det_data = detections_dict[image_idx]["detections"][det_idx]
                
        # get the key use by sapiens and deduce the number of keypoints and the model
        det_keys = list(det_data.keys())
        
        if det_data["validity"] == "valid":
            det_keys_sapiens = [key for key in det_keys if key.startswith("scores_sapiens")] # scores_sapiens_308_03b for example (scores_sapiens_NKPTS_MODEL)
            if len(det_keys_sapiens) == 0:
                # no sapiens keypoints found, lets use 0s
                sapiens_keypoints = np.zeros((308, 2)) # shape (308, 2)
                sapiens_keypoints_scores = np.zeros(308) # shape (308,)
            else:
                sapiens_model = det_keys_sapiens[0].split("_")[-1] # if several possibility always take the first one
                sapiens_num_keypoints = int(det_keys_sapiens[0].split("_")[2]) # if several possibility always take the first one
                assert sapiens_num_keypoints == 308, "Sapiens number of keypoints {} expected to be 308".format(sapiens_num_keypoints)
                assert sapiens_model == "06b" or sapiens_model == "03b" or sapiens_model == "1b", "Sapiens model {} expected to be 06b, 03b or 1b".format(sapiens_model)
                sapiens_keypoints = np.array(det_data[f"keypoints_sapiens_{sapiens_num_keypoints}_{sapiens_model}"]).reshape(-1, 2) # shape (308, 2)
                sapiens_keypoints_scores = np.array(det_data[f"scores_sapiens_{sapiens_num_keypoints}_{sapiens_model}"]) # shape (308,)
                assert sapiens_keypoints.shape[0] == 308, "Sapiens keypoint have 308 keypoints ({} vs 308)".format(sapiens_keypoints.shape[0])
                assert sapiens_keypoints_scores.shape[0] == 308, "Sapiens keypoints have 308 scores ({} vs 308)".format(sapiens_keypoints_scores.shape[0])
        else:
            sapiens_keypoints = np.zeros((308, 2)) # shape (308, 2)
            sapiens_keypoints_scores = np.zeros(308) # shape (308,)
        
        current_engagement = track_info["engagements"][idx_in_track]
        
        track_data["image_file"].append(image_file)
        track_data["image_index"].append(image_idx)
        track_data["mask_rle"].append(det_data["mask_rle"])
        track_data["mask_size"].append(det_data["mask_size"])
        track_data["validity"].append(det_data["validity"])
        track_data["current_segment"].append(current_segment_idx)
        track_data["total_segments"].append(len(valid_segments))
        track_data["position_in_segment"].append(position_in_segment)
        track_data["length_of_current_segment"].append(length_of_current_segment)
        track_data["timestamp"].append(timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"))
        track_data["timestamp_sec"].append(timestamp_sec)
        track_data["timestamp_track"].append(time_since_track_first_image_sec)
        track_data["engagement"].append(current_engagement)
        track_data["time_to_first_interaction"].append(time_to_first_interaction_sec)
        track_data["xmin"].append(np.round(det_data["xmin"], 2))
        track_data["ymin"].append(np.round(det_data["ymin"], 2))
        track_data["xmax"].append(np.round(det_data["xmax"], 2))
        track_data["ymax"].append(np.round(det_data["ymax"], 2))
        
        vitpose_keypoints_coords = np.array(det_data["keypoints_vitpose"]).reshape(-1, 2) # shape (17, 2)
        vitpose_keypoints_scores = np.array(det_data["scores_vitpose"]) # shape (17,)
        
        sapiens_keypoints_coords = sapiens_keypoints # shape (308, 2)
        sapiens_keypoints_scores = sapiens_keypoints_scores # shape (308,)
        
        for vitpose_keypoint_name, vitpose_keypoint_score, vitpose_keypoint_coords in zip(VITPOSE_KEYPOINTS_NAMES, vitpose_keypoints_scores, vitpose_keypoints_coords):
            track_data[vitpose_keypoint_name+"_x"].append(np.round(vitpose_keypoint_coords[0], 2))
            track_data[vitpose_keypoint_name+"_y"].append(np.round(vitpose_keypoint_coords[1], 2))
            track_data[vitpose_keypoint_name+"_score"].append(np.round(vitpose_keypoint_score, 2))

        for sapiens_keypoint_name, sapiens_keypoint_score, sapiens_keypoint_coords in zip(SAPIENS_KEYPOINTS_NAMES, sapiens_keypoints_scores, sapiens_keypoints_coords):
            track_data[sapiens_keypoint_name+"_x"].append(np.round(sapiens_keypoint_coords[0], 2))
            track_data[sapiens_keypoint_name+"_y"].append(np.round(sapiens_keypoint_coords[1], 2))
            track_data[sapiens_keypoint_name+"_score"].append(np.round(sapiens_keypoint_score, 2))

    track_df = pd.DataFrame(track_data)

    return track_df

def get_first_interaction_index(track_info):
    """ Get the first interaction index of the track, -1 if no interaction

    Interaction is defined as engagement level == 3
    
    To find the first interaction index, we only consider the elements in track_info["engagements"], to get the absolute index (in the episode) use the track_info["images_indexes"]
    
    Args:
        track_info (dict): track info

    Returns:
        int: first interaction index (among the number of elements in track_info)
    """
    # if engages at any time, we only consider the part of the track before the first engagement
    engagement_values = track_info["engagements"]
    is_engaged_mask = np.array([engagement == 3 for engagement in engagement_values]) # 3 means interacting
    if np.sum(is_engaged_mask) != 0:
        first_interaction_index = np.where(is_engaged_mask)[0][0]
    else:
        first_interaction_index = -1
    
    return first_interaction_index
    

def is_track_valid(track_info, 
                   min_number_of_valid_detections, 
                   min_number_of_quality_detections, 
                   quality_threshold = 0.5, 
                   contiguous_tolerance = 0, # unused
                   ) -> list:
    """ Check if the track is valid

    We define n_valid_conditions = min_number_of_quality_detections = min_number_of_valid_detections except if one of them is 0, then depending on the case it will be min_number_of_quality_detections or min_number_of_valid_detections
    
    We create a mask with AND for the conditions on validity and quality (except if one of them is 0) and check if the sum of that mask is greater than or equal to n_valid_conditions
    We extend the mask True areas by contiguous_tolerance in both directions
    Then the track is valid if at least one contiguous true segment is at least n_valid_conditions long
    
    If the track interacts at any time, we only consider the part of the track before the first interaction

    Args:
        track_info (dict): track info (engagements, validities, qualities)
        min_number_of_valid_detections (int): minimum number of valid detections
        min_number_of_quality_detections (int): minimum number of quality detections
        quality_threshold (float): quality threshold
        contiguous_tolerance (int): contiguous tolerance (both directions) (UNUSED)

    Returns:
        list: list of contiguous true segments if the track is valid, empty list otherwise
    """
    
    track_info = track_info.copy()
    
    # if interacts at any time, we only consider the part of the track before the first interaction
    engagement_values = track_info["engagements"]
    is_engaged_mask = np.array([engagement == 3 for engagement in engagement_values]) # 3 means interacting
    if np.sum(is_engaged_mask) != 0:
        first_interaction_index = np.where(is_engaged_mask)[0][0]
        track_info["images_indexes"] = track_info["images_indexes"][:first_interaction_index]
        track_info["det_indexes"] = track_info["det_indexes"][:first_interaction_index]
        track_info["validities"] = track_info["validities"][:first_interaction_index]
        track_info["qualities"] = track_info["qualities"][:first_interaction_index]
        track_info["engagements"] = track_info["engagements"][:first_interaction_index]

    if track_info["images_indexes"] == []:
        return []

    # create masks based on the image indexes (to avoid gaps)
    min_image_index = min(track_info["images_indexes"])
    max_image_index = max(track_info["images_indexes"])
    track_len = max_image_index - min_image_index + 1
        
    if track_len < min_number_of_valid_detections or track_len < min_number_of_quality_detections:
        return []
    
    is_valid_mask = np.zeros(track_len, dtype=bool) # array of shape (track_len,) with track_len the number of elements between the first and last image index of the track
    is_quality_mask = np.zeros(track_len, dtype=bool) # array of shape (track_len,) with track_len the number of elements between the first and last image index of the track
    
    for image_idx, validity, quality in zip(track_info["images_indexes"], track_info["validities"], track_info["qualities"]):
        relative_image_idx = image_idx - min_image_index
        is_valid_mask[relative_image_idx] = validity == "valid"
        is_quality_mask[relative_image_idx] = quality >= quality_threshold
    
    # is_valid_mask = np.array([validity == "valid" for validity in track_info["validities"]]) # array of shape (track_len,)
    # is_quality_mask = np.array([quality >= quality_threshold for quality in track_info["qualities"]]) # array of shape (track_len,)
    
    if min_number_of_quality_detections == 0:
        n_valid_conditions = min_number_of_valid_detections
        conditions_mask = is_valid_mask # array of shape (track_len,)
    elif min_number_of_valid_detections == 0:
        n_valid_conditions = min_number_of_quality_detections
        conditions_mask = is_quality_mask # array of shape (track_len,)
    else:
        n_valid_conditions = min_number_of_valid_detections # should be equal to min_number_of_quality_detections
        conditions_mask = np.logical_and(is_valid_mask, is_quality_mask) # array of shape (track_len,)

    if np.sum(conditions_mask) < n_valid_conditions:
        # not enough valid detections anyway
        return []


    # # first dilate by tolerance
    # Removed tolerance because we want to have data for all elements in a segment
    # if contiguous_tolerance > 0:
    #     for i in range(conditions_mask.shape[0]):
    #         if conditions_mask[i]:
    #             conditions_mask[max(0, i-contiguous_tolerance):min(track_len, i+contiguous_tolerance)] = True

    contiguous_true_segments = get_contiguous_true_segments(conditions_mask) # list of segments [(start, length), ...] in image indexes, relative to the first image index of the track
    contiguous_true_segments = [(int(start), int(length)) for start, length in contiguous_true_segments] # in image indexes, relative to the first image index of the track
    
    is_valid = False
    for segment in contiguous_true_segments:
        if segment[1] >= n_valid_conditions:
            is_valid = True
            break
    
    if is_valid:
        return contiguous_true_segments
    else:
        return []
    
def get_episode_data(args, dataset_recording_dir, annotations_recording_dir, ep):
    """ List tracks that meet a minimum of validity criteria
    and pass the raw data from detections to get_track_data, then concat all the dicts 

    Args:
        args (argparse.Namespace): Arguments
        dataset_recording_dir (str): Path to the dataset recording directory (for getting recording name)
        annotations_recording_dir (str): Path to the annotations recording directory (contains episodes with detections)
        ep (str): Episode name

    Returns:
        pd.DataFrame: DataFrame for all the tracks of the episode with each line corresponding to one timestep and the columns being features (keypoints, box coordinates, scores, etc.)
    """

    episode_dir = os.path.join(annotations_recording_dir, "episodes", ep)
    assert os.path.isdir(episode_dir), "Episode directory does not exist at {}".format(episode_dir)
    assert os.path.exists(os.path.join(episode_dir, "annotations_360.json")), "Detections file does not exist at {}".format(os.path.join(episode_dir, "annotations_360.json"))

    detections_path = os.path.join(episode_dir, "annotations_360.json")
    detections_dict = read_json_to_dic(detections_path)

    # Now investigate at the track level
    tracks_infos = get_tracks_infos(detections_dict, engagement_key="engagement_level")
    try:
        image_shape = get_image_shape(detections_dict)
    except:
        # unfortunately some older detections files do not have the "image_shape" key, so we use the default shape
        image_shape = (1920, 3840) # default shape

    tracks_dfs_list = []
    for track_id, track_info in tracks_infos.items():
        valid_segments = is_track_valid(track_info, min_number_of_valid_detections=30, min_number_of_quality_detections=30, quality_threshold=0.5, contiguous_tolerance=1)  # the list of contiguous valid segments of the form [(start,length), (start,length), ...] in image indexes, relative to the first image index of the track if the track is valid, empty list otherwise
        if len(valid_segments) > 0:
            # get the data from the track using the detections_dict and the images_idxs in track_info
            track_first_interaction_index = get_first_interaction_index(track_info)
            track_df = get_track_data(detections_dict, track_info, track_first_interaction_index, valid_segments)
            # add a column at the beginning of the dataframe with the track_id
            track_df.insert(0, "track_id", track_id)
            
            unique_track_identifier = f"{os.path.basename(dataset_recording_dir)}_{ep}_{track_id}"
            track_df.insert(0, "unique_track_identifier", unique_track_identifier)
            track_df.insert(0, "image_width", image_shape[1])
            track_df.insert(0, "image_height", image_shape[0])
            track_df.insert(0, "episode", ep)
            
            tracks_dfs_list.append(track_df.copy())
    
    tracks_df = pd.concat(tracks_dfs_list) if len(tracks_dfs_list) > 0 else pd.DataFrame()

    return tracks_df.copy()

def get_recording_data(args, dataset_recording_dir, annotations_recording_dir=None):
    """ For each episode call get_episode_data and concatenate the data of the episodes

    Args:
        args (argparse.Namespace): Arguments
        dataset_recording_dir (str): Path to the dataset recording directory (contains episodes_flag.json)
        annotations_recording_dir (str): Path to the annotations recording directory (contains episodes with detections), defaults to dataset_recording_dir if None

    Returns:
        pd.DataFrame: DataFrame for all the episodes of the recording with each line corresponding to one timestep and the columns being features (keypoints, box coordinates, scores, etc.)
    """
    
    if annotations_recording_dir is None:
        annotations_recording_dir = dataset_recording_dir
    
    episodes_flags_path = os.path.join(dataset_recording_dir, "episodes_flag.json")
    assert os.path.exists(episodes_flags_path), "Episodes flags file does not exist at {}".format(episodes_flags_path)
    episodes_flags = read_json_to_dic(episodes_flags_path)
    
    episodes_dfs_list = []
    for episode_name, episode_data in episodes_flags.items():
        valid_flags = is_episode_usable(args, episode_data["flag"])
        if not valid_flags:
            continue
        
        episode_df = get_episode_data(args, dataset_recording_dir, annotations_recording_dir, episode_name)
        prSuccess(f"[{dataset_recording_dir}][{episode_name}] Got episode_df of length {len(episode_df)}")
        if len(episode_df) > 0:
            # sometimes not a single track is valid, so we need to check
            episode_df.insert(0, "recording", os.path.basename(dataset_recording_dir))
            episodes_dfs_list.append(episode_df.copy())
        else:
            pass
            # prWarning("No valid tracks found in episode {}, will not export it".format(episode_name))
    
    if len(episodes_dfs_list) == 0:
        prWarning("No valid episodes found in recording {}, will not export it, all tracks, all episodes were invalid".format(os.path.basename(dataset_recording_dir)))
    
    episodes_df = pd.concat(episodes_dfs_list) if len(episodes_dfs_list) > 0 else pd.DataFrame()
    
    return episodes_df

def check_recording_validity_wrapper(args_and_dirs):
    """Wrapper function for multiprocessing check_recording_validity"""
    args, dataset_recording_dir, annotations_recording_dir = args_and_dirs
    return check_recording_validity(args, dataset_recording_dir, annotations_recording_dir)

def get_recording_data_wrapper(args_and_dirs):
    """Wrapper function for multiprocessing get_recording_data"""
    args, dataset_recording_dir, annotations_recording_dir = args_and_dirs
    return get_recording_data(args, dataset_recording_dir, annotations_recording_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./output/")
    parser.add_argument("--input", "-i", type=str, default="images_360", help="Input type (only supports images_360)")
    parser.add_argument("--recordings", "-r", type=str, default="all", help="Recordings to export, all or a list of recordings names")
    parser.add_argument("--allow_empty_dirs", "-ed", action="store_true", default=False, help="Allow empty directories to be processed")
    parser.add_argument("--splitted_dataset", "-sd", action="store_true", default=False, help="Use splitted dataset mode, the data and the annotations are separated into different directories (default: %(default)s)")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of workers to use")
    args = parser.parse_args()

    if args.workers == 0:
        args.workers = mp.cpu_count() // 2
    prInfo(f"Using {args.workers} workers")
    
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
    

    
    # validity check first - parallelized
    prInfo(f"Checking {len(recordings_list)} recordings validity (using {args.workers} workers)")
    recording_args_list = []
    for recording in recordings_list:
        if args.splitted_dataset:
            assert os.path.isdir(os.path.join(args.dir, "dataset", recording)), "Dataset directory {} should exist and be a directory".format(os.path.join(args.dir, "dataset", recording))
            dataset_recording_dir = os.path.join(args.dir, "dataset", recording)
            assert os.path.isdir(os.path.join(args.dir, "annotations", recording)), "Annotations directory {} should exist and be a directory".format(os.path.join(args.dir, "annotations", recording))
            annotations_recording_dir = os.path.join(args.dir, "annotations", recording)
        else:
            dataset_recording_dir = os.path.join(args.dir, recording)
            assert os.path.isdir(dataset_recording_dir), "Recording {} should exist and be a directory".format(dataset_recording_dir)
            annotations_recording_dir = None
        recording_args_list.append((args, dataset_recording_dir, annotations_recording_dir))
    
    # Check validity in parallel
    with mp.Pool(processes=args.workers) as pool:
        validity_results = pool.map(check_recording_validity_wrapper, recording_args_list)
    
    # Only extract tracks if there is at least one valid episode
    valid_recordings = []
    valid_recording_args_list = []
    for recording, n_valid_episodes, (args, dataset_recording_dir, annotations_recording_dir) in zip(recordings_list, validity_results, recording_args_list):
        if n_valid_episodes > 0:
            prSuccess(f"Recording {recording} is valid ({n_valid_episodes} episodes are valid) and will be exported later")
            valid_recordings.append(recording)
            valid_recording_args_list.append((args, dataset_recording_dir, annotations_recording_dir))
        else:
            prWarning(f"Recording {recording} has no valid episodes and will be skipped")
    
    if len(valid_recordings) == 0:
        prWarning("No valid recordings found, exiting...")
        exit(1)
    
    # Get recording data (i.e. extract tracks)
    prInfo("Getting recording data in parallel...")
                
    with mp.Pool(processes=args.workers) as pool:
    # with mp.Pool(processes=1) as pool:
        # Get recording data
        recording_data_results = pool.map(get_recording_data_wrapper, valid_recording_args_list)
        
    # Process results
    recordings_dfs_list = []
    
    for recording, recording_df in zip(valid_recordings, recording_data_results):
        prSuccess(f"Got data for recording {recording} (entries : {len(recording_df)}, unique tracks : {len(recording_df['unique_track_identifier'].unique()) if len(recording_df) > 0 else 0})")
        recordings_dfs_list.append(recording_df)

    assert len(recordings_dfs_list) > 0, "No recordings found or all invalid"
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dataset_name = f"exported_dataset_{args.input}_{timestamp}"
    # save the dataset in a directory
    dirname = args.dir.split("/")[-1] if args.dir.split("/")[-1] != "" else args.dir.split("/")[-2]
    dataset_dir = os.path.join(args.dir, "..", dirname+"_exported_datasets", dataset_name)
    if not os.path.exists(os.path.join(args.dir, "..", dirname+"_exported_datasets")):
        prInfo("Creating exported datasets directory at absolute path {}".format(os.path.join(args.dir, "..", dirname+"_exported_datasets")))
        os.makedirs(os.path.join(args.dir, "..", dirname+"_exported_datasets"))
    os.makedirs(dataset_dir, exist_ok=True)
    
    for count, recording_df in enumerate(recordings_dfs_list):
        recording_name = recording_df["recording"].unique()[0]
        save_recording_name = f"ssupaug-{count:04d}-of-{len(recordings_dfs_list):04d}.csv"
        recording_df.to_csv(os.path.join(dataset_dir, save_recording_name), index=False)
        prSuccess(f"Saved recording {recording_name} to {save_recording_name}")

    # dataset_path = os.path.join(dataset_dir, f"dataset.csv")
    # recordings_df.to_csv(dataset_path, index=False)
    # prSuccess(f"Exported data to {dataset_path} (entries : {len(recordings_df)}, unique tracks : {len(recordings_df['unique_track_identifier'].unique())})")

    prSuccess(f"Exported dataset to {dataset_dir}")