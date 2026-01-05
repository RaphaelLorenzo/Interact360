
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
import argparse

DATA_DIRECTORY_EPISODES = None
OUTPUT_DIRECTORY = None

def create_video_from_images(images_dir, output_dir, output_filename, recording_name, episode_name):
    """Create a video from images using ffmpeg and move it to the output directory.
    
    Args:
        images_dir: Directory containing the images
        output_dir: Directory where the video should be saved
        output_filename: Name of the output video file
        recording_name: Name of the recording (for logging)
        episode_name: Name of the episode (for logging)
    """
    final_video_path = os.path.join(output_dir, output_filename)
    
    # Skip if video already exists
    if os.path.exists(final_video_path):
        return
    
    # prInfo(f"Creating video from {images_dir} for {recording_name}/{episode_name}")
    # temp output in parent directory of images_dir
    temp_video_path = os.path.join(os.path.dirname(images_dir), "temp_output.mp4")
    
    if os.path.exists(temp_video_path):
        # delete it
        os.remove(temp_video_path)
        prWarning(f"Deleted existing temp video: {temp_video_path}")
    
    # Change to the images directory for glob pattern to work correctly
    original_cwd = os.getcwd()
    os.chdir(images_dir)
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", "15",
        "-pattern_type", "glob",
        "-i", "*.jpg",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        temp_video_path
    ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    os.chdir(original_cwd)
    
    if result.returncode == 0:
        prSuccess(f"Successfully created video: {temp_video_path}")
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        # Move video to output directory
        shutil.move(temp_video_path, final_video_path)
        prSuccess(f"Video moved to: {final_video_path}")
    else:
        prError(f"Failed to create video for {recording_name}/{episode_name}")
        prError(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

def process_episode(args_tuple):
    """Process a single episode: export filenames and create videos.
    
    Args:
        args_tuple: Tuple containing (recording_name, episode_name)
    
    Returns:
        recording_name, episode_name: For tracking completion
    """
    recording_name, episode_name = args_tuple
    
    prInfo(f"[{recording_name}/episode {episode_name}] Processing...")
    tic = time.time()
    
    images_360_dir = os.path.join(DATA_DIRECTORY_EPISODES, recording_name, "episodes", episode_name, "images_360")
    miniatures_360_dir = os.path.join(DATA_DIRECTORY_EPISODES, recording_name, "episodes", episode_name, "miniatures_images_360_scale_4")
    assert(os.path.exists(images_360_dir)), f"Images 360 directory does not exist: {images_360_dir}"
    assert(os.path.exists(miniatures_360_dir)), f"Miniatures 360 directory does not exist: {miniatures_360_dir}"
    
    images_360_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_360_dir)) for f in fn if f.endswith(".jpg")]
    images_360_list.sort()
    miniatures_360_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(miniatures_360_dir)) for f in fn if f.endswith(".jpg")]
    miniatures_360_list.sort()
    assert(len(images_360_list) == len(miniatures_360_list)), f"Number of images and miniatures must be the same, got {len(images_360_list)} and {len(miniatures_360_list)} in {recording_name}/{episode_name}"
    
    images_360_basenames = [os.path.basename(image) for image in images_360_list]
    
    # Create output directory structure
    output_recording_dir = os.path.join(OUTPUT_DIRECTORY, "recordings", recording_name)
    output_episode_dir = os.path.join(output_recording_dir, "episodes", episode_name)
    os.makedirs(output_episode_dir, exist_ok=True)
    
    # Export basenames to text file
    filenames_txt_path = os.path.join(output_episode_dir, "images_360_filenames.txt")
    with open(filenames_txt_path, "w") as f:
        for basename in images_360_basenames:
            f.write(basename + "\n")
    # prSuccess(f"Exported {len(images_360_basenames)} filenames to {filenames_txt_path}")
    
    # Create video from full images
    create_video_from_images(images_360_dir, output_episode_dir, "images_360_full.mp4", recording_name, episode_name)
    
    # Create video from miniature images
    # create_video_from_images(miniatures_360_dir, output_episode_dir, "images_360_mini.mp4", recording_name, episode_name)
    
    toc = time.time()
    prSuccess(f"[{recording_name}/episode {episode_name}] Processed in {toc - tic:.2f} seconds ({len(images_360_basenames)} frames)")
    
    return recording_name, episode_name, len(images_360_basenames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", "-dd", type=str, default="/media/raphael/hucedisk/ShelfyInteractFull/data_extractor/", help="Dataset directory containing the episodes (default: %(default)s)")
    parser.add_argument("--output_directory", "-od", type=str, default="/media/raphael/hucedisk3/HUI360-Videos/", help="Output directory for the videos (default: %(default)s)")
    args = parser.parse_args()
    
    DATA_DIRECTORY_EPISODES = args.dataset_directory
    OUTPUT_DIRECTORY = args.output_directory
    
    assert(os.path.exists(DATA_DIRECTORY_EPISODES)), f"Dataset directory does not exist: {DATA_DIRECTORY_EPISODES}"
    assert(os.path.exists(OUTPUT_DIRECTORY)), f"Output directory does not exist: {OUTPUT_DIRECTORY}"
        
    recordings_list_episodes_json = os.listdir(DATA_DIRECTORY_EPISODES)
    recordings_list_episodes_json.sort()

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, "recordings"))

    UNEXPORTED_RECORDINGS = ["rosbag2_2025_10_07-16_00_15"]

    # Prepare all episodes for processing
    all_episodes = []
    for recording_name in recordings_list_episodes_json:
        
        if recording_name in UNEXPORTED_RECORDINGS:
            #or ("ShelfyInteractFull" in DATA_DIRECTORY_EPISODES and (recording_name.startswith("rosbag2_2025_07") or recording_name.startswith("rosbag2_2025_10_17") or recording_name.startswith("rosbag2_2025_10_20"))):
            prWarning(f"Skipping unexported recording: {recording_name}")
            continue
        
        # episodes_flags_json_path = os.path.join(DATA_DIRECTORY_EPISODES, recording_name, "episodes_flag.json")
        # episodes_flags_json_dict = read_json_to_dic(episodes_flags_json_path)
        episodes_json_path = os.path.join(DATA_DIRECTORY_EPISODES, recording_name, "episodes.json")
        episodes_json_dict = read_json_to_dic(episodes_json_path)

        for idx, ep in enumerate(episodes_json_dict):
            all_episodes.append((recording_name, f"{idx:04d}"))

    # Process episodes in parallel using half the available cores
    num_workers = 1 # max(1, mp.cpu_count() // 2)
    num_workers = min(num_workers, len(all_episodes))
    prInfo(f"Processing {len(all_episodes)} episodes using {num_workers} workers")

    if num_workers > 1:
        pool = None
        try:
            pool = mp.Pool(processes=num_workers)
            results = pool.map(process_episode, all_episodes)
            prSuccess(f"Successfully processed {len(results)} episodes with {sum([result[2] for result in results])} frames")
        except KeyboardInterrupt:
            prWarning("Keyboard interrupt received. Shutting down pool gracefully...")
            if pool is not None:
                pool.terminate()
                pool.join()
            prWarning("Pool terminated. Exiting.")
            sys.exit(0)
        except Exception as e:
            prError(f"Error during parallel processing: {e}")
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    else:
        prInfo(f"Processing {len(all_episodes)} episodes sequentially")
        total_frames = 0
        for episode in all_episodes:
            recording_name, episode_name, frames = process_episode(episode)
            total_frames += frames
        prSuccess(f"Successfully processed {len(all_episodes)} episodes with {total_frames} frames")
