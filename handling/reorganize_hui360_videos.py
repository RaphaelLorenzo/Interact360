
import os
import sys
import shutil
import subprocess
import time
import multiprocessing as mp
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *
import socket
hostname = socket.gethostname()
if hostname == "raphael-Precision-3591":
    OUTPUT_DIRECTORY = "/media/raphael/hucedisk3/HUI360-Videos/"
else:
    raise NotImplementedError("Not implemented")
    OUTPUT_DIRECTORY = "/home/rlorenzo/group_storage/rlorenzo/HUI360-Videos/"


def reorganize_episode(args_tuple):
    """Process a single episode: export filenames and create videos.
    
    Args:
        args_tuple: Tuple containing (recording_name, episode_name)
    
    Returns:
        recording_name, episode_name: For tracking completion
    """
    recording_name, episode_name = args_tuple
    
    prInfo(f"[{recording_name}/episode {episode_name}] Checking and moving...")
    
    episode_dir = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name, "episodes", episode_name)
    
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "dataset", recording_name, episode_name, "images_360"), exist_ok=True)
    
    filenames_txt_path = os.path.join(episode_dir, "images_360_filenames.txt")
    assert(os.path.exists(filenames_txt_path)), f"Filenames text file does not exist: {filenames_txt_path}"
    shutil.move(filenames_txt_path, os.path.join(OUTPUT_DIRECTORY, "annotations", recording_name, episode_name, "images_360_filenames.txt"))

    mini_vid_path = os.path.join(episode_dir, "images_360_mini.mp4")
    assert(os.path.exists(mini_vid_path)), f"Mini video does not exist: {mini_vid_path}"
    shutil.move(mini_vid_path, os.path.join(OUTPUT_DIRECTORY, "video_mini", recording_name, episode_name, "images_360_mini.mp4"))
    
    full_vid_path = os.path.join(episode_dir, "images_360_full.mp4")
    assert(os.path.exists(full_vid_path)), f"Full video does not exist: {full_vid_path}"
    shutil.move(full_vid_path, os.path.join(OUTPUT_DIRECTORY, "video_full", recording_name, episode_name, "images_360_full.mp4"))
    
    annotations_path = os.path.join(episode_dir, "annotations_360.json")
    assert(os.path.exists(annotations_path)), f"Annotations file does not exist: {annotations_path}"
    shutil.move(annotations_path, os.path.join(OUTPUT_DIRECTORY, "annotations", recording_name, episode_name, "annotations_360.json"))
    
    if "astor" in recording_name or "albee" in recording_name:
        interaction_zone_tracking_path = os.path.join(episode_dir, "interaction_zone_tracking_360.json")
        assert(os.path.exists(interaction_zone_tracking_path)), f"Interaction zone tracking file does not exist: {interaction_zone_tracking_path}"
        shutil.move(interaction_zone_tracking_path, os.path.join(OUTPUT_DIRECTORY, "annotations", recording_name, episode_name, "interaction_zone_tracking_360.json"))

    return recording_name, episode_name

recordings_list_episodes_json = os.listdir(os.path.join(OUTPUT_DIRECTORY, "recordings"))
recordings_list_episodes_json.sort()

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "recordings"))


os.makedirs(os.path.join(OUTPUT_DIRECTORY, "dataset"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIRECTORY, "video_mini"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIRECTORY, "video_full"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIRECTORY, "annotations"), exist_ok=True)

for recording_name in recordings_list_episodes_json:
    
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "dataset", recording_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "video_mini", recording_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "video_full", recording_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "annotations", recording_name), exist_ok=True)

    episodes_json_path = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name, "episodes.json")
    episodes_flags_json_path = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name, "episodes_flag.json")
    episodes_flags_json_dict = read_json_to_dic(episodes_flags_json_path)

    shutil.move(episodes_json_path, os.path.join(OUTPUT_DIRECTORY, "dataset", recording_name, "episodes.json"))
    shutil.move(episodes_flags_json_path, os.path.join(OUTPUT_DIRECTORY, "dataset", recording_name, "episodes_flag.json"))
    
    
    for episode_name, flags_dict in episodes_flags_json_dict.items():
        
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, "dataset", recording_name, episode_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, "video_mini", recording_name, episode_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, "video_full", recording_name, episode_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, "annotations", recording_name, episode_name), exist_ok=True)
        
        # recording_name, episode_name = reorganize_episode((recording_name, episode_name))

# episodes_to_fix = []

# Prepare all episodes for processing
# all_episodes = []
# for recording_name in recordings_list_episodes_json:

#     episodes_flags_json_path = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name, "episodes_flag.json")
#     episodes_flags_json_dict = read_json_to_dic(episodes_flags_json_path)

#     for episode_name, flags_dict in episodes_flags_json_dict.items():
#         # all_episodes.append((recording_name, episode_name, flags_dict))
#         if "GDPR" in flags_dict["flag"]:
#             print(recording_name, episode_name, flags_dict)
#             episodes_to_fix.append((recording_name, episode_name))
            
# # Recreate mini videos from full videos where mini has error but full doesn't

# if len(episodes_to_fix) > 0:
#     prInfo(f"Found {len(episodes_to_fix)} episodes to fix (recreating mini videos from full videos, to be blacked out)")
    
#     for recording_name, episode_name in episodes_to_fix:
#         # Parse recording_name and episode_name from episode['name'] (format: "recording_name/episode episode_name")
        
#         episode_dir = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name, "episodes", episode_name)
#         full_vid_path = os.path.join(episode_dir, "images_360_full.mp4")
#         mini_vid_path = os.path.join(episode_dir, "images_360_mini.mp4")
        
#         if not os.path.exists(full_vid_path):
#             prError(f"Full video does not exist: {full_vid_path}")
#             continue
        
#         prInfo(f"Recreating mini video for {recording_name}/episode {episode_name}...")
        
#         # Use ffmpeg to downscale the full video by 4
#         ffmpeg_cmd = [
#             "ffmpeg",
#             "-i", full_vid_path,
#             "-vf", "scale=iw/4:ih/4",
#             "-y",  # Overwrite output file if it exists
#             mini_vid_path
#         ]
        
#         try:
#             result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
#             if result.returncode == 0:
#                 prSuccess(f"Successfully recreated mini video for {recording_name}/episode {episode_name}")
#             else:
#                 prError(f"Failed to recreate mini video for {recording_name}/episode {episode_name}:\n{result.stderr}")
#         except Exception as e:
#             prError(f"Exception while recreating mini video for {recording_name}/episode {episode_name}: {e}")
# else:
#     prInfo("No episodes need mini video recreation")


