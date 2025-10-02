import os
import sys
import subprocess
import shutil
import time
import multiprocessing as mp

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *

from tqdm import tqdm
import argparse

def process_video(args_tuple):
    """Worker function to extract images from a single video file
    
    Args:
        args_tuple: A tuple containing the video path, index of the video (among the multiple videos processed in parallel), output path, and framerate
        
    Returns:
        idx: The index of the video
        video_path: The original path to the video
    """
    video_path, idx, output_path, framerate = args_tuple
    
    # create an output subdirectory
    output_subdir = os.path.join(output_path, f"{idx:03d}")
    os.makedirs(output_subdir, exist_ok=True)
    
    prInfo(f"Processing video {idx}: {video_path}")
    
    try:
        # extract the images
        result = subprocess.run(["ffmpeg", "-i", video_path, "-filter:v", "fps={}".format(args.framerate), "-q:v", "1", os.path.join(output_subdir, "frame_%06d.jpg")], capture_output=True, text=True)
        
        if result.returncode == 0:
            prSuccess(f"Successfully processed video {idx}: {video_path}")
        else:
            prError(f"Failed to process video {idx}: {video_path}")
            prError(f"Error: {result.stderr}")
            
    except Exception as e:
        prError(f"Exception processing video {idx}: {video_path} - {str(e)}")
    
    return idx, video_path


def main(args):
    video_name = os.path.basename(args.video_path).split(".")[0]
    video_extension = os.path.basename(args.video_path).split(".")[1]
    prInfo(f"Video name: {video_name}")
    prInfo(f"Video extension: {video_extension}")
    
    output_dir = os.path.join(args.save_dir, video_name, "episodes")
    output_path = os.path.join(output_dir, "images_360") #, "0000", "images_360")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        prSuccess(f"Output path {output_path} created")
    else:
        prWarning(f"Output path {output_path} already exists, will overwrite it")
        time.sleep(1)
        shutil.rmtree(output_path)
        prInfo(f"Output path {output_path} removed")
    
    output_txt_path = os.path.join(args.save_dir, video_name, 'metadata_360_extraction.txt')
    with open(output_txt_path, "w") as f:
        f.write(args.video_path + "\n")
        f.write(str(args.framerate) + "\n")
    
    prInfo(f"Running ffmpeg for {args.video_path} to split into multiple videos of 5 minutes each to allow for parallel processing later")
    
    # split the mkv file in videos of 5 minutes each for parallel processing
    output_path_splitted_videos = os.path.join(args.save_dir, video_name, 'splitted_videos_360')
    if not os.path.exists(output_path_splitted_videos):
        os.makedirs(output_path_splitted_videos)
    else:
        # remove it
        prWarning(f"Output path {output_path_splitted_videos} already exists, will overwrite it")
        time.sleep(1)
        shutil.rmtree(output_path_splitted_videos)
        os.makedirs(output_path_splitted_videos)
    
    # split the mkv file in videos of 5 minutes each for parallel processing
    prInfo(f"Splitting {args.video_path} in {output_path_splitted_videos}...")
    mkvfile_rn = os.path.basename(args.video_path).split(".")[0]
    subprocess.run(["ffmpeg", "-i", args.video_path, "-c", "copy", "-map", "0", "-f", "segment", "-segment_time", "300", "-reset_timestamps", "1", os.path.join(output_path_splitted_videos, f"{mkvfile_rn}_%03d.{video_extension}")])
    
    # list the splitted videos
    splitted_videos = [os.path.join(output_path_splitted_videos, f) for f in os.listdir(output_path_splitted_videos) if f.endswith(f".{video_extension}")]
    splitted_videos.sort()
    
    # process the videos in parallel
    prInfo(f"Processing {len(splitted_videos)} videos in parallel")
    num_workers = args.workers if args.workers > 0 else int(mp.cpu_count() / 2)
    num_workers = min(num_workers, len(splitted_videos))
    prInfo(f"Using {num_workers} workers for parallel processing of {len(splitted_videos)} videos")
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_video, [(video, idx, output_path, args.framerate) for idx, video in enumerate(splitted_videos)]),
            total=len(splitted_videos),
            desc="Extracting images from videos"
        ))

    prSuccess(f"Completed extracting images from {len(results)} videos")
    
    # move the images to the output path
    subdirs = [f"{output_path}/{i:03d}" for i in range(len(splitted_videos))]
    image_index = 0
    for subdir in subdirs:
        # move all images  from subdir to output_path by increasing the image_index
        images_in_subdir = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".jpg")]
        images_in_subdir.sort()
        for image in images_in_subdir:
            shutil.move(image, os.path.join(output_path, f"frame_{image_index:06d}.jpg"))
            image_index += 1
        
        # remove the subdir when it is emptied
        shutil.rmtree(subdir)
        prSuccess(f"Removed temporary subdirectory {subdir}")
    
    prSuccess(f"Completed moving all {image_index} images to {output_path}")
    
    # remove splitted videos
    shutil.rmtree(output_path_splitted_videos)
    prSuccess(f"Removed temporary splitted videos directory {output_path_splitted_videos}")
    
    # Create episodes bases on the rules
    if args.episodes_method == "single":
        os.makedirs(os.path.join(output_dir, "episodes", "0000"))
        # move the whole images_360 directory to the episode directory
        shutil.move(output_path, os.path.join(output_dir, "episodes", "0000", "images_360"))
        
    elif args.episodes_method == "time":
        # list all files in output_path
        files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".jpg")]
        files.sort()
        # create episodes based on time
        episode_duration_in_frame = int(args.framerate * args.max_episodes_time)
        n_episodes = (len(files) // episode_duration_in_frame) + 1
        prInfo(f"Creating {n_episodes} episodes based on time (total frames : {len(files)}, frames per episode : {episode_duration_in_frame})")
        for i in range(n_episodes):
            if not os.path.exists(os.path.join(output_dir, "episodes", f"{i:04d}", "images_360")):
                os.makedirs(os.path.join(output_dir, "episodes", f"{i:04d}", "images_360"))
            else:
                prWarning(f"Episode {i:04d} already exists, will remove it")
                time.sleep(1)
                shutil.rmtree(os.path.join(output_dir, "episodes", f"{i:04d}", "images_360"))
                os.makedirs(os.path.join(output_dir, "episodes", f"{i:04d}", "images_360"))
                
            files_in_episode = files[i*episode_duration_in_frame:(i+1)*episode_duration_in_frame]
            for file in files_in_episode:
                shutil.move(file, os.path.join(output_dir, "episodes", f"{i:04d}", "images_360", os.path.basename(file)))
            prSuccess(f"Created episode {i:04d} with {len(files_in_episode)} frames")
            
    elif args.episodes_method == "detection":
        # create episodes based on detection
        raise NotImplementedError("Not implemented")

    shutil.rmtree(output_path)
    prSuccess(f"Removed temporary images directory {output_path}")
    
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", "-v", type=str, default="./demo/data/2025_06_13_11_15_31_Theta360.mkv")
    parser.add_argument("--save_dir", "-s", type=str, default="./demo/output")
    parser.add_argument("--framerate", "-f", type=int, default=15)
    parser.add_argument("--workers", "-w", type=int, default=0, help="Number of cores to use for parallel processing. Default: 0 (use half of available cores)") 
    parser.add_argument("--episodes_method", "-em", type=str, default="single", help="Method to use to create episodes : single, time, detection")
    parser.add_argument("--max_episodes_time", "-met", type=int, default=300, help="Max time in seconds to create episodes, does not apply if method is single")
    args = parser.parse_args()
    
    if args.episodes_method == "single":
        prInfo(f"Create a single episodes : all images in /episodes/0000/images_360")
        prWarning(f"Ignoring max time per episode")
    elif args.episodes_method == "time":
        prInfo(f"Create episodes based on time : max {args.max_episodes_time} seconds per episode")
    elif args.episodes_method == "detection":
        prInfo(f"Create episodes based on detection : with max {args.max_episodes_time} seconds per episode")
    else:
        prError(f"Invalid episodes method : {args.episodes_method}")
        
    assert(os.path.exists(args.video_path)), f"The video path does not exist at {args.video_path}"
    assert(os.path.isdir(args.save_dir)), f"The save directory does not exist at {args.save_dir}"
    
    # extract the images
    main(args)