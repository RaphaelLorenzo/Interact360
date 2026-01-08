# This script creates a lightweight version of the dataset CSVs
# by removing sapiens keypoints and mask_rle columns

import argparse
import os
import sys
import pandas as pd
import multiprocessing as mp
from glob import glob

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))

from misc.print_utils import *


def get_columns_to_drop(columns: list) -> list:
    """Get list of columns to drop (those starting with 'sapiens_' or 'mask_rle')
    
    Args:
        columns: List of column names
        
    Returns:
        List of column names to drop
    """
    return [col for col in columns if col.startswith('sapiens_') or col.startswith('mask_rle')]


def process_csv(input_path: str, output_path: str) -> tuple:
    """Process a single CSV file, removing heavy columns
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        
    Returns:
        Tuple of (original_columns, remaining_columns, dropped_columns)
    """
    df = pd.read_csv(input_path)
    original_cols = list(df.columns)
    
    cols_to_drop = get_columns_to_drop(original_cols)
    df_light = df.drop(columns=cols_to_drop)
    
    df_light.to_csv(output_path, index=False)
    
    return len(original_cols), len(df_light.columns), len(cols_to_drop)


def process_csv_wrapper(args_tuple):
    """Wrapper function for multiprocessing"""
    idx, total, input_path, output_path = args_tuple
    filename = os.path.basename(input_path)
    n_original, n_remaining, n_dropped = process_csv(input_path, output_path)
    output_filename = os.path.basename(output_path)
    prSuccess(f"[{idx+1}/{total}] {filename} -> {output_filename} (dropped {n_dropped} columns, {n_remaining} remaining)")
    return n_original, n_remaining, n_dropped


def main():
    parser = argparse.ArgumentParser(description="Create lightweight CSV files by removing sapiens and mask_rle columns")
    parser.add_argument("--input_dir", "-i", type=str, required=True, 
                        help="Input directory containing CSV files")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory (default: input_dir + '_light')")
    parser.add_argument("--only_new_version", "-nv", action="store_true", default=False,
                        help="Only process new version of the dataset (default: %(default)s)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of workers for parallel processing (0 = half of CPU count)")
    args = parser.parse_args()
    
    if args.workers == 0:
        args.workers = mp.cpu_count() // 2
    prInfo(f"Using {args.workers} workers")
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        prError(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        # Create sibling directory with _light suffix
        input_dir_clean = args.input_dir.rstrip('/')
        args.output_dir = input_dir_clean + '_light'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    prInfo(f"Output directory: {args.output_dir}")
    
    # Find all CSV files in input directory
    csv_files = glob(os.path.join(args.input_dir, "*.csv"))
    csv_files.sort()
    
    if args.only_new_version:
        csv_files = [f for f in csv_files if ("astor" in f) or ("albee" in f) or ("rosbag2" in f)]
    
    if len(csv_files) == 0:
        prWarning(f"No CSV files found in {args.input_dir}")
        sys.exit(1)
    
    prInfo(f"Found {len(csv_files)} CSV files to process")
    
    # Build task list
    tasks = []
    for i, csv_path in enumerate(csv_files):
        filename = os.path.basename(csv_path)
        output_filename = filename.replace('.csv', '_light.csv')
        output_path = os.path.join(args.output_dir, output_filename)
        tasks.append((i, len(csv_files), csv_path, output_path))
    
    # Process CSV files in parallel
    total_original = 0
    total_remaining = 0
    total_dropped = 0
    
    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(process_csv_wrapper, tasks)
        for n_original, n_remaining, n_dropped in results:
            total_original = n_original
            total_remaining = n_remaining
            total_dropped = n_dropped
    else:
        for task in tasks:
            n_original, n_remaining, n_dropped = process_csv_wrapper(task)
            total_original = n_original
            total_remaining = n_remaining
            total_dropped = n_dropped

    prSuccess(f"\nDone! Processed {len(csv_files)} files")
    prInfo(f"Column reduction: {total_original} -> {total_remaining} ({total_dropped} columns removed)")
    prInfo(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

