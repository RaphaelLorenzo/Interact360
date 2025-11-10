# This script aims at exporting a report about the content of the raw data of each recording session (at the episode level)

import argparse
import os
from re import TEMPLATE
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

import pandas as pd
import shutil
from misc.print_utils import *
from misc.other_utils import *
from export.export_utils import get_tracks_infos, is_episode_usable, get_contiguous_true_segments

# for each episode in the report we want (for images and images_360):
# - number of images
# - number of detections
# - number of images with at least one detection
# - number of detections with a quality score >= 0.5
# - number of tracks
# - number of tracks with at least 1 valid detection
# - number of tracks with more than 30 valid detections
# - number of tracks with more than 30 valid detections that have a quality score >= 0.5
# - number of tracks with more than 30 contiguous valid detections that have a quality score >= 0.5
# - number of tracks with more than 120 valid detections
# - number of tracks with more than 120 valid detections that have a quality score >= 0.5
# - number of tracks with more than 120 contiguous valid detections that have a quality score >= 0.5
# - number of segments of 30 contiguous valid detections that have a quality score >= 0.5
# - number of non overlapping subsets of 30 contiguous valid detections that have a quality score >= 0.5
# - number of interactions with at least 1 valid detection
# - number of interactions with at least 30 valid detections
# - number of interactions with at least 30 valid detections that have a quality score >= 0.5
# - number of interactions with at least 30 contiguous valid detections that have a quality score >= 0.5
# - number of interactions with at least 120 valid detections
# - number of interactions with at least 120 valid detections that have a quality score >= 0.5
# - number of interactions with at least 120 contiguous valid detections that have a quality score >= 0.5

from export.export_utils import RECORDING_TO_PLACE

def count_tracks_with_conditions(tracks_infos, 
                                 min_number_of_valid_detections, 
                                 min_number_of_quality_detections, 
                                 quality_threshold = 0.5, 
                                 contiguous_tolerance = 0, # unused
                                 before_engagement_only = False):
    """ Count the number of tracks with the given conditions 
    
    We define n_valid_conditions = min_number_of_quality_detections = min_number_of_valid_detections except if one of them is 0, then depending on the case it will be min_number_of_quality_detections or min_number_of_valid_detections
    
    We create a mask with AND for the conditions on validity and quality (except if one of them is 0) and check if the sum of that mask is greater than or equal to n_valid_conditions
    For each track, we extend the mask True areas by contiguous_tolerance in both directions
    Then we count the contiguouse True segments that are at least n_valid_conditions long
    Then we count the number of non overlapping subsets of length n_valid_conditions in the True segments

    So, min_number_of_valid_detections must be equal to min_number_of_quality_detections (and it defines the enough condition), or one of them must be 0 (then we define enough as min_number_of_quality_detections or min_number_of_valid_detections depending on the case)
    
    Args:
        tracks_infos (dict): tracks infos
        min_number_of_valid_detections (int): minimum number of valid detections (if 0 ignore this condition)
        min_number_of_quality_detections (int): minimum number of quality detections (with a quality score >= quality_threshold) (if 0 ignore this condition)
        quality_threshold (float): quality threshold for the detections
        contiguous_tolerance (int): tolerance for contiguous valid detections (UNUSED)
        before_engagement_only (bool): if True, only consider the tracks that are engaged and only consider them before the first engagement

    Returns:
        int: number of tracks that meet the conditions
        int: number of tracks with at least one contiguous true segment that is at least n_valid_conditions long
        int: total number across all tracks of contiguous true segments that are at least n_valid_conditions long
        int: total number across all tracks of non overlapping subsets of length n_valid_conditions in the True segments
    """
    
    if min_number_of_valid_detections != min_number_of_quality_detections and min_number_of_valid_detections != 0 and min_number_of_quality_detections != 0:
        raise ValueError("min_number_of_valid_detections and min_number_of_quality_detections must be equal, or one of them must be 0")
    

    tracks_count = 0
    tracks_contiguous_count = 0
    segments_count = 0
    subsets_count = 0
    for track_id, track_info_data in tracks_infos.items():
        track_info = track_info_data.copy()
        if before_engagement_only:
            engagement_values = track_info["engagements"]
            is_engaged_mask = np.array([engagement == 3 for engagement in engagement_values]) # 3 means interacting
            if np.sum(is_engaged_mask) == 0:
                continue
            first_engagement_index = np.where(is_engaged_mask)[0][0]
            track_info["images_indexes"] = track_info["images_indexes"][:first_engagement_index]
            track_info["det_indexes"] = track_info["det_indexes"][:first_engagement_index]
            track_info["validities"] = track_info["validities"][:first_engagement_index]
            track_info["qualities"] = track_info["qualities"][:first_engagement_index]
            track_info["engagements"] = track_info["engagements"][:first_engagement_index]
        
        if track_info["images_indexes"] == []:
            continue
        
        # create masks based on the image indexes (to avoid gaps)
        min_image_index = min(track_info["images_indexes"])
        max_image_index = max(track_info["images_indexes"])
        track_len = max_image_index - min_image_index + 1
        
        if track_len < min_number_of_valid_detections or track_len < min_number_of_quality_detections:
            continue

        is_valid_mask = np.zeros(track_len, dtype=bool)
        is_quality_mask = np.zeros(track_len, dtype=bool)
        
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
    
        if np.sum(conditions_mask) == 0:
            # no valid detections
            continue
            
        if np.sum(conditions_mask) >= n_valid_conditions:
            tracks_count += 1
    
        # now the segment related counting
        # # first dilate by tolerance
        # Removed tolerance because we want to have data for all elements in a segment
        # if contiguous_tolerance > 0:
        #     for i in range(conditions_mask.shape[0]):
        #         if conditions_mask[i]:
        #             conditions_mask[max(0, i-contiguous_tolerance):min(track_len, i+contiguous_tolerance)] = True
    
        contiguous_true_segments = get_contiguous_true_segments(conditions_mask) # list of segments [(start, length), ...] in image indexes, relative to the first image index of the track
        
        # count the number of segments that are long enough
        valid_contiguous_segments_for_track = 0
        for segment in contiguous_true_segments:
            if segment[1] >= n_valid_conditions:
                valid_contiguous_segments_for_track += 1
        
        if valid_contiguous_segments_for_track > 0:
            tracks_contiguous_count += 1
        
        # add them to the total
        segments_count += valid_contiguous_segments_for_track
        
        # for each segment see how many subsets we can do
        valid_subsets_for_track = 0
        for segment in contiguous_true_segments:
            if segment[1] >= n_valid_conditions:
                valid_subsets_for_track += segment[1] // n_valid_conditions
        
        # add them to the total
        subsets_count += valid_subsets_for_track
            
    return tracks_count, tracks_contiguous_count, segments_count, subsets_count

def make_recording_report(args, recording_dir):
    episodes_flags_path = os.path.join(recording_dir, "episodes_flag.json")
    
    #### FOR DEBUG
    # if not os.path.exists(episodes_flags_path):
    #     EPISODES_POSSILBLE_LOC = os.path.join("/media/raphael/hucedisk3/ShelfyDatasetNameNextExtension/data_extractor", os.path.basename(recording_dir), "episodes_flag.json")
    #     EPISODES_POSSILBLE_LOC1 = os.path.join("/media/raphael/hucedisk/ShelfyDatasetNameExtension/data_extractor", os.path.basename(recording_dir), "episodes_flag.json")
    #     EPISODES_POSSILBLE_LOC2 = os.path.join("/media/raphael/hucedisk/ShelfyDatasetName/data_extractor", os.path.basename(recording_dir), "episodes_flag.json")
    #     if os.path.exists(EPISODES_POSSILBLE_LOC):
    #         # copy to expected location
    #         shutil.copy(EPISODES_POSSILBLE_LOC, episodes_flags_path)
    #         prDebug("Copied episodes flags file to expected location at {} from {}".format(episodes_flags_path, EPISODES_POSSILBLE_LOC))
            
    #     elif os.path.exists(EPISODES_POSSILBLE_LOC1):
    #         # copy to expected location
    #         shutil.copy(EPISODES_POSSILBLE_LOC1, episodes_flags_path)
    #         prDebug("Copied episodes flags file to expected location at {} from {}".format(episodes_flags_path, EPISODES_POSSILBLE_LOC1))
            
    #     elif os.path.exists(EPISODES_POSSILBLE_LOC2):
    #         # copy to expected location
    #         shutil.copy(EPISODES_POSSILBLE_LOC2, episodes_flags_path)
    #         prDebug("Copied episodes flags file to expected location at {} from {}".format(episodes_flags_path, EPISODES_POSSILBLE_LOC2))
            
    #     else:
    #         raise ValueError("Episodes flags file does not exist at {} nor at {} nor at {}".format(episodes_flags_path, EPISODES_POSSILBLE_LOC, EPISODES_POSSILBLE_LOC1, EPISODES_POSSILBLE_LOC2))
        
    assert os.path.exists(episodes_flags_path), "Episodes flags file does not exist at {}".format(episodes_flags_path)
    episodes_flags = read_json_to_dic(episodes_flags_path)
    
    episodes_reports = {}
    
    for episode_name, episode_data in episodes_flags.items():
        valid_flags = is_episode_usable(args, episode_data["flag"])
        
        # check that we have the necessary inputs in the episode
        episode_dir = os.path.join(recording_dir, "episodes", episode_name)
        
        # # check that the detections are valid
        # # detections_path = os.path.join(episode_dir, "detections_sam2.json")
        # detections_path = os.path.join(episode_dir, "sam2_detections_360.json")
        # detections_dict = read_json_to_dic(detections_path)
        # # total number of detections
        # total_detections = 0
        # for element in detections_dict:
        #     total_detections += len(element["detections"])
        # images_dir = os.path.join(episode_dir, "images")
        # images_files = os.listdir(images_dir)
        # images_files = [os.path.join(images_dir, f) for f in images_files if f.endswith(".jpg")]
        # images_files.sort()
        # assert len(detections_dict) > 0, "No detections found in {}".format(detections_path)
        # if len(images_files) == 0 and args.allow_empty_dirs:
        #     # allow empty dirs to work only with labels
        #     pass
        # else:
        #     assert len(detections_dict) == len(images_files), "Number of detections and images should be the same ({} vs {})".format(len(detections_dict), len(images_files))
        
        # same for 360
        # detections_path_360 = os.path.join(episode_dir, "detections_sam2_360.json")
        detections_path_360 = os.path.join(episode_dir, "sam2_detections_360.json")
        try:
            detections_dict_360 = read_json_to_dic(detections_path_360)
        except Exception as e:
            # potentially corrupted file, skip it
            prError(f"Error reading detections file {detections_path_360}: {e}")
            continue
        
        # total number of detections
        total_detections_360 = 0
        for element in detections_dict_360:
            total_detections_360 += len(element["detections"])
            
        if not args.allow_empty_dirs:
            images_dir_360 = os.path.join(episode_dir, "images_360")
            images_files_360 = os.listdir(images_dir_360)
            images_files_360 = [os.path.join(images_dir_360, f) for f in images_files_360 if f.endswith(".jpg")]
            images_files_360.sort()
        else:
            images_files_360 = []

        assert len(detections_dict_360) > 0, "No detections found in {}".format(detections_path_360)

        if len(images_files_360) == 0 and args.allow_empty_dirs:
            # allow empty dirs to work only with labels
            pass
        else:
            assert len(detections_dict_360) == len(images_files_360), "Number of detections and images should be the same ({} vs {})".format(len(detections_dict_360), len(images_files_360))


        total_dets = [sum(len(element["detections"]) for element in detections_dict_360)]


        prInfo(f"[{os.path.basename(recording_dir)}][{episode_name} : {valid_flags}] {len(detections_dict_360)} images 360 | {total_dets} total detections")
        # Now investigate at the track level
        # tracks_infos = get_tracks_infos(detections_dict, engagement_key="engagement_level_auto_1", allow_skip = not valid_flags)
        tracks_infos_360 = get_tracks_infos(detections_dict_360, engagement_key="engagement_level", allow_skip = not valid_flags)

        # basic minimal conditions
        # tracks_count_1_valid, _, _, _ = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=1, min_number_of_quality_detections=0)
        tracks_count_1_valid_360, _, _, _ = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=1, min_number_of_quality_detections=0)
        # tracks_count_1_valid_before_engagement, _, _, _ = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=1, min_number_of_quality_detections=0, before_engagement_only=True)
        tracks_count_1_valid_before_engagement_360, _, _, _ = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=1, min_number_of_quality_detections=0, before_engagement_only=True)

        # tracks_count_30_valid, _, _, _ = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=30, min_number_of_quality_detections=0)
        tracks_count_30_valid_360, _, _, _ = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=30, min_number_of_quality_detections=0)
        # tracks_count_30_valid_before_engagement, _, _, _ = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=30, min_number_of_quality_detections=0, before_engagement_only=True)
        tracks_count_30_valid_before_engagement_360, _, _, _ = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=30, min_number_of_quality_detections=0, before_engagement_only=True)

        # more advanced conditions + count contiguous
        # allow for an extension of 1 in the contiguous tolerance
        # tracks_count_30_valid_quality, tracks_count_30_valid_quality_contiguous, segments_count_30_valid_quality, subsets_count_30_valid_quality = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=30, min_number_of_quality_detections=30, contiguous_tolerance=1)
        tracks_count_30_valid_quality_360, tracks_count_30_valid_quality_contiguous_360, segments_count_30_valid_quality_360, subsets_count_30_valid_quality_360 = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=30, min_number_of_quality_detections=30, contiguous_tolerance=1)
        # tracks_count_30_valid_quality_before_engagement, tracks_count_30_valid_quality_contiguous_before_engagement, segments_count_30_valid_quality_before_engagement, subsets_count_30_valid_quality_before_engagement = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=30, min_number_of_quality_detections=30, contiguous_tolerance=1, before_engagement_only=True)
        tracks_count_30_valid_quality_before_engagement_360, tracks_count_30_valid_quality_contiguous_before_engagement_360, segments_count_30_valid_quality_before_engagement_360, subsets_count_30_valid_quality_before_engagement_360 = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=30, min_number_of_quality_detections=30, contiguous_tolerance=1, before_engagement_only=True)
        
        # tracks_count_120_valid_quality, tracks_count_120_valid_quality_contiguous, segments_count_120_valid_quality, subsets_count_120_valid_quality = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=120, min_number_of_quality_detections=120, contiguous_tolerance=1)
        tracks_count_120_valid_quality_360, tracks_count_120_valid_quality_contiguous_360, segments_count_120_valid_quality_360, subsets_count_120_valid_quality_360 = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=120, min_number_of_quality_detections=120, contiguous_tolerance=1)
        # tracks_count_120_valid_quality_before_engagement, tracks_count_120_valid_quality_contiguous_before_engagement, segments_count_120_valid_quality_before_engagement, subsets_count_120_valid_quality_before_engagement = count_tracks_with_conditions(tracks_infos, min_number_of_valid_detections=120, min_number_of_quality_detections=120, contiguous_tolerance=1, before_engagement_only=True)
        tracks_count_120_valid_quality_before_engagement_360, tracks_count_120_valid_quality_contiguous_before_engagement_360, segments_count_120_valid_quality_before_engagement_360, subsets_count_120_valid_quality_before_engagement_360 = count_tracks_with_conditions(tracks_infos_360, min_number_of_valid_detections=120, min_number_of_quality_detections=120, contiguous_tolerance=1, before_engagement_only=True)
        
        
        # output dict
        episode_report = {
            
            "flags": episode_data["flag"],
            "valid_flags": valid_flags,
            
            # Main images

            # # images and detections counts
            # "nimages": len(detections_dict),
            # "ndetections": total_detections,

            # # tracks counting with just the valid conditions
            # "tracks_count_1_valid": tracks_count_1_valid,
            # "tracks_count_30_valid": tracks_count_30_valid,
            
            # # ... also considering the quality
            # "tracks_count_30_valid_quality": tracks_count_30_valid_quality,
            # "tracks_count_120_valid_quality": tracks_count_120_valid_quality,
            
            # # ... and with at least one contiguous segment
            # "tracks_count_30contiguous_valid_quality": tracks_count_30_valid_quality_contiguous,
            # "tracks_count_120contiguous_valid_quality": tracks_count_120_valid_quality_contiguous,
            
            # # ... and the total count of segments and subsets
            # "segments_count_30_valid_quality": segments_count_30_valid_quality,
            # "segments_count_120_valid_quality": segments_count_120_valid_quality,
            # "subsets_count_30_valid_quality": subsets_count_30_valid_quality,
            # "subsets_count_120_valid_quality": subsets_count_120_valid_quality,
            
            # # ... and before engagement only
            # "tracks_count_1_valid_before_engagement": tracks_count_1_valid_before_engagement,
            # "tracks_count_30_valid_before_engagement": tracks_count_30_valid_before_engagement,
            # "tracks_count_30_valid_quality_before_engagement": tracks_count_30_valid_quality_before_engagement,
            # "tracks_count_120_valid_quality_before_engagement": tracks_count_120_valid_quality_before_engagement,
            # "tracks_count_30contiguous_valid_quality_before_engagement": tracks_count_30_valid_quality_contiguous_before_engagement,
            # "tracks_count_120contiguous_valid_quality_before_engagement": tracks_count_120_valid_quality_contiguous_before_engagement,
            # "segments_count_30_valid_quality_before_engagement": segments_count_30_valid_quality_before_engagement,
            # "segments_count_120_valid_quality_before_engagement": segments_count_120_valid_quality_before_engagement,
            # "subsets_count_30_valid_quality_before_engagement": subsets_count_30_valid_quality_before_engagement,
            # "subsets_count_120_valid_quality_before_engagement": subsets_count_120_valid_quality_before_engagement,
            
            # 360 images
            # images and detections counts
            "ndetections_360": total_detections_360,
            "nimages_360": len(detections_dict_360),

            # tracks counting with just the valid conditions
            "tracks_count_1_valid_360": tracks_count_1_valid_360,
            "tracks_count_30_valid_360": tracks_count_30_valid_360,
            "tracks_count_30_valid_quality_360": tracks_count_30_valid_quality_360,
            "tracks_count_120_valid_quality_360": tracks_count_120_valid_quality_360,
            
            # ... also considering the quality
            "tracks_count_30contiguous_valid_quality_360": tracks_count_30_valid_quality_contiguous_360,
            "tracks_count_120contiguous_valid_quality_360": tracks_count_120_valid_quality_contiguous_360,
            
            # ... and with at least one contiguous segment
            "tracks_count_30contiguous_valid_quality_360": tracks_count_30_valid_quality_contiguous_360,
            "tracks_count_120contiguous_valid_quality_360": tracks_count_120_valid_quality_contiguous_360,
            
            # ... and the total count of segments and subsets
            "segments_count_30_valid_quality_360": segments_count_30_valid_quality_360,
            "segments_count_120_valid_quality_360": segments_count_120_valid_quality_360,
            "subsets_count_30_valid_quality_360": subsets_count_30_valid_quality_360,
            "subsets_count_120_valid_quality_360": subsets_count_120_valid_quality_360,
            
            # ... and before engagement only
            "tracks_count_1_valid_before_engagement_360": tracks_count_1_valid_before_engagement_360,   
            "tracks_count_30_valid_before_engagement_360": tracks_count_30_valid_before_engagement_360,
            "tracks_count_30_valid_quality_before_engagement_360": tracks_count_30_valid_quality_before_engagement_360,
            "tracks_count_120_valid_quality_before_engagement_360": tracks_count_120_valid_quality_before_engagement_360,
            "tracks_count_30contiguous_valid_quality_before_engagement_360": tracks_count_30_valid_quality_contiguous_before_engagement_360,
            "tracks_count_120contiguous_valid_quality_before_engagement_360": tracks_count_120_valid_quality_contiguous_before_engagement_360,
            "segments_count_30_valid_quality_before_engagement_360": segments_count_30_valid_quality_before_engagement_360,
            "segments_count_120_valid_quality_before_engagement_360": segments_count_120_valid_quality_before_engagement_360,
            "subsets_count_30_valid_quality_before_engagement_360": subsets_count_30_valid_quality_before_engagement_360,
            "subsets_count_120_valid_quality_before_engagement_360": subsets_count_120_valid_quality_before_engagement_360,
            
        }
        
        episodes_reports[episode_name] = episode_report
    
    return episodes_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./output/")
    parser.add_argument("--recordings", "-r", type=str, default="all", help="Recordings to export, all or a list of recordings names")
    parser.add_argument("--allow_empty_dirs", "-ed", action="store_true", default=False, help="Allow empty directories to be processed")
    args = parser.parse_args()
    
    assert os.path.isdir(args.dir), "Directory should exist at {}".format(args.dir)
    
    if args.recordings == "all":
        # only keep directories
        recordings_list = [f for f in os.listdir(os.path.join(args.dir)) if os.path.isdir(os.path.join(args.dir, f))]
    else:
        recordings_list = args.recordings.split(",")
        
    recordings_list.sort()
    
    df_elements = []
    for recording in recordings_list:
        assert os.path.isdir(os.path.join(args.dir, recording)), "Recording {} should exist and be a directory".format(recording)
        episodes_reports = make_recording_report(args, os.path.join(args.dir, recording)) # trigger assertions from inside the function in case of problems
        for ep, ep_report in episodes_reports.items():
            place = RECORDING_TO_PLACE[recording] if recording in RECORDING_TO_PLACE else "Unknown"
            df_elements.append({
                "recording": recording,
                "episode": ep,
                "place": place,
                **ep_report
            })
    
    df = pd.DataFrame(df_elements)
    
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    report_path = os.path.join(args.dir, "..", "dataset_reports", "dataset_report_{}.csv".format(timestamp))
    if not os.path.exists(os.path.join(args.dir, "..", "dataset_reports")):
        prInfo("Creating dataset reports directory at absolute path {}".format(os.path.join(args.dir, "..", "dataset_reports")))
        os.makedirs(os.path.join(args.dir, "..", "dataset_reports"))
    df.to_csv(report_path, index=False)
    prInfo("Dataset report saved at {}".format(report_path))
        