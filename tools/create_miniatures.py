import argparse
import os
import subprocess
import multiprocessing as mp

import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *
from misc.other_utils import *

import PIL

from tqdm import tqdm


def process_single_image(input_path: str, output_path: str, scale_factor: int, force: bool = False) -> bool:
    """
    Downsample a single image using ffmpeg.
    
    Args:
        input_path (str): path to the input image
        output_path (str): path to the output image
        scale_factor (int): scale factor for the miniatures
        force (bool): force re-run the miniatures creation even if the output already exists
        
    Returns:
        bool: True if the image was processed successfully or already exists, False otherwise
    """
    # Skip if output already exists
    if os.path.exists(output_path) and not force:
        return True
    
    # Build ffmpeg command for downsampling
    # Using scale filter: scale=iw/scale_factor:ih/scale_factor
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale=iw/{scale_factor}:ih/{scale_factor}',
        '-q:v', '2',  # High quality JPEG compression
        '-y',  # Overwrite output files without asking
        output_path
    ]
    
    try:
        # Run ffmpeg command with suppressed output
        result = subprocess.run(
            ffmpeg_cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True
        
    except subprocess.CalledProcessError as e:
        prError(f"Failed to process {os.path.basename(input_path)}: {e}")
        prError(f"FFmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        prError("FFmpeg not found. Please install ffmpeg to use this functionality.")
        return False



def process_image_directory(args, ep, source_dir: str, target_dir: str, scale_factor: int, num_workers: int = 1):
    """
    Process a full image directory using ffmpeg for downsampling with multiprocessing.
    
    Args:
        args (argparse.Namespace): arguments from the command line
        ep (str): episode name to be processed
        source_dir (str): source directory name (images to be miniaturized)
        target_dir (str): target directory name (where to save the miniatures)
        scale_factor (int): scale factor for the miniatures
        num_workers (int): number of workers to use for the multiprocessing        
    """
    image_dir = os.path.join(args.dir, "episodes", ep, source_dir)
    save_dir = os.path.join(args.dir, "episodes", ep, target_dir)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        prInfo(f"Created directory: {save_dir}")
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        prWarning(f"No images found in {image_dir}")
        return
    
    image_files.sort()
    prInfo(f"Processing {len(image_files)} images from {source_dir} with scale factor 1/{scale_factor} using {num_workers} workers")
    
    # Prepare tasks for multiprocessing
    tasks = []
    for image_file in image_files:
        input_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(save_dir, image_file)
        tasks.append((input_path, output_path, scale_factor, args.force))
    
    # Process images with multiprocessing
    if num_workers == 1:
        # Sequential processing
        for task in tqdm(tasks, desc=f"Creating miniatures for {source_dir}"):
            process_single_image(*task)
    else:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            pool.starmap(process_single_image, tasks)
    
    prSuccess(f"Completed processing {source_dir} -> {target_dir}")



def create_miniatures(args, ep):
    """
    Create miniature versions of images using ffmpeg for downsampling.
    Processes images, images_360, and images_external directories.
    """
    
    # Process 360 images directory
    if os.path.exists(os.path.join(args.dir, "episodes", ep, "images_360")):
        process_image_directory(
            args, ep, 
            source_dir="images_360", 
            target_dir="miniatures_images_360_scale_{}".format(args.scale_360), 
            scale_factor=args.scale_360,
            num_workers=args.workers
        )
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./output/rosbags_2025_03_26_15_25_25")
    parser.add_argument("--episodes", "-e", type=str, default="all")
    parser.add_argument("--scale_360", "-s3", type=int, default=4)
    parser.add_argument("--workers", "-w", type=int, default=0)
    parser.add_argument("--force", "-f", action="store_true", default=False, help="Force re-run the miniatures creation even if the output already exists")

    args = parser.parse_args()
    
    assert os.path.isdir(args.dir), "Directory should exist"
    assert os.path.isdir(os.path.join(args.dir, "episodes")), "Episodes directory should exist"
    
    # Determine number of workers (same logic as extract_tracks_images.py)
    if args.workers == 0:
        args.workers = mp.cpu_count() // 2
    prInfo(f"Using {args.workers} workers")
    
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()
    
    for ep in episodes_list:
        assert os.path.isdir(os.path.join(args.dir, "episodes", ep)), "Episode {} should exist".format(ep)

    prInfo("Creating miniatures for episodes : {}".format(", ".join(episodes_list)))
    
    for ep in episodes_list:
        create_miniatures(args, ep)
    