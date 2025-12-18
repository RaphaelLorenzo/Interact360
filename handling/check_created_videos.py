
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
    DATA_DIRECTORY_EPISODES = "/media/raphael/hucedisk/ShelfyInteractFull/data_extractor/"
    # DATA_DIRECTORY_EPISODES = "/media/raphael/hucedisk3/ShelfyDatasetNameNextExtension/data_extractor/"
    OUTPUT_DIRECTORY = "/media/raphael/hucedisk3/HUI360-Videos/"
else:
    DATA_DIRECTORY_EPISODES = "/home/rlorenzo/group_storage/rlorenzo/SSUP/FullVideos/extracted"
    OUTPUT_DIRECTORY = "/home/rlorenzo/group_storage/rlorenzo/HUI360-Videos/"


def verify_episode(args_tuple):
    """Process a single episode: export filenames and create videos.
    
    Args:
        args_tuple: Tuple containing (recording_name, episode_name)
    
    Returns:
        recording_name, episode_name: For tracking completion
    """
    recording_name, episode_name = args_tuple
    
    prInfo(f"[{recording_name}/episode {episode_name}] Checking...")
    
    episode_dir = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name, "episodes", episode_name)
    if not os.path.exists(episode_dir):
        prError(f"Episode directory does not exist: {episode_dir}")
        return recording_name, episode_name, False, False, False
    
    filenames_txt_path = os.path.join(episode_dir, "images_360_filenames.txt")
    if not os.path.exists(filenames_txt_path):
        prError(f"Filenames text file does not exist: {filenames_txt_path}")
        return recording_name, episode_name, False, False, False
    
    with open(filenames_txt_path, "r") as f:
        images_360_basenames = f.readlines()
    images_360_basenames = [basename.strip() for basename in images_360_basenames]
    
    mini_video_error = False
    mini_vid_path = os.path.join(episode_dir, "images_360_mini.mp4")
    if os.path.exists(mini_vid_path):
        # Check number of frames in the mini video if it exists
        # Use ffprobe to get the frame count
        ffprobe_cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames",
                        "-of", "default=nokey=1:noprint_wrappers=1",
                        mini_vid_path
                    ]
        try:
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                nb_frames_str = result.stdout.strip()
                try:
                    nb_frames = int(nb_frames_str)
                except Exception:
                    prWarning(f"Could not interpret nb_frames ({nb_frames_str}) for {mini_vid_path}")
                    mini_video_error = True
                    nb_frames = None

                if nb_frames is not None:
                    if nb_frames != len(images_360_basenames):
                        prError(f"Mini video frame count ({nb_frames}) does not match number of images ({len(images_360_basenames)}) in {mini_vid_path}")
                        mini_video_error = True
                else:
                    prWarning(f"Could not determine frame count for mini video: {mini_vid_path}")
                    mini_video_error = True
            else:
                prWarning(f"ffprobe failed for mini video: {mini_vid_path}\n{result.stderr}")
                mini_video_error = True
        except Exception as e:
            prWarning(f"Exception during ffprobe frame check for mini video: {mini_vid_path} | {e}")
            mini_video_error = True
            
        prSuccess(f"[{recording_name}/episode {episode_name}] | video mini | Verified (nb frames: {nb_frames})")
    
    full_video_error = False
    full_vid_path = os.path.join(episode_dir, "images_360_full.mp4")
    if os.path.exists(full_vid_path):
        ffprobe_cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames",
                        "-of", "default=nokey=1:noprint_wrappers=1",
                        full_vid_path
                    ]
        try:
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                nb_frames_str = result.stdout.strip()
                try:
                    nb_frames = int(nb_frames_str)
                except Exception:
                    prWarning(f"Could not interpret nb_frames ({nb_frames_str}) for {full_vid_path}")
                    full_video_error = True
                    nb_frames = None

                if nb_frames is not None:
                    if nb_frames != len(images_360_basenames):
                        prError(f"Full video frame count ({nb_frames}) does not match number of images ({len(images_360_basenames)}) in {full_vid_path}")
                        full_video_error = True
                else:
                    prWarning(f"Could not determine frame count for full video: {full_vid_path}")
                    full_video_error = True
            else:
                prWarning(f"ffprobe failed for full video: {full_vid_path}\n{result.stderr}")
                full_video_error = True
        except Exception as e:
            prWarning(f"Exception during ffprobe frame check for full video: {full_vid_path} | {e}")
            full_video_error = False
        prSuccess(f"[{recording_name}/episode {episode_name}] | video full | Verified (nb frames: {nb_frames})")


    prSuccess(f"[{recording_name}/episode {episode_name}] | All | Verified (nb frames: {len(images_360_basenames)})")
    
    return recording_name, episode_name, True, mini_video_error, full_video_error

recordings_list_episodes_json = os.listdir(DATA_DIRECTORY_EPISODES)
recordings_list_episodes_json.sort()

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, "recordings"))

UNEXPORTED_RECORDINGS = ["rosbag2_2025_10_07-16_00_15"]

# Prepare all episodes for processing
all_episodes = []
for recording_name in recordings_list_episodes_json:
    
    if recording_name in UNEXPORTED_RECORDINGS:
        prWarning(f"Skipping unexported recording: {recording_name}")
        continue

    episodes_json_path = os.path.join(DATA_DIRECTORY_EPISODES, recording_name, "episodes.json")
    episodes_json_dict = read_json_to_dic(episodes_json_path)

    for idx, ep in enumerate(episodes_json_dict):
        all_episodes.append((recording_name, f"{idx:04d}"))

errors_episodes = []
for episode in all_episodes:
    recording_name, episode_name, success, mini_video_error, full_video_error = verify_episode(episode)
    if not success or mini_video_error or full_video_error:
        errors_episodes.append({"name": f"{recording_name}/episode {episode_name}", "success": success, "mini_video_error": mini_video_error, "full_video_error": full_video_error})

prSuccess(f"Successfully checked {len(all_episodes)} episodes")
prSuccess(f"Found {len(errors_episodes)} errors")
for episode in errors_episodes:
    prError(f"Error: {episode}")
    
# write errors to a txt report
timestamp = time.strftime("%Y%m%d_%H%M%S")
with open(f"errors_report_{timestamp}.txt", "w") as f:
    for episode in errors_episodes:
        f.write(f"{episode['name']}\n")
        f.write(f"Success: {episode['success']}\n")
        f.write(f"Mini video error: {episode['mini_video_error']}\n")
        f.write(f"Full video error: {episode['full_video_error']}\n")
        f.write("\n")
