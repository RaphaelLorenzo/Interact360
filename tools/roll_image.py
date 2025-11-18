#!/usr/bin/env python3
"""
Script to roll a panoptic image by a specified number of pixels.
"""

import argparse
import numpy as np
from PIL import Image
import os


def roll_image(input_path, output_path, roll_x=0, roll_y=0):
    """
    Load an image, roll it by specified pixels, and save it.
    
    Args:
        input_path: Path to input image
        output_path: Path to save rolled image
        roll_x: Number of pixels to roll horizontally (positive = right, negative = left)
        roll_y: Number of pixels to roll vertically (positive = down, negative = up)
    """
    # Load the image
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # Roll the image
    rolled_array = np.roll(img_array, shift=(roll_y, roll_x), axis=(0, 1))
    
    # Convert back to PIL Image and save
    rolled_img = Image.fromarray(rolled_array)
    rolled_img.save(output_path)
    
    print(f"Image rolled by ({roll_x}, {roll_y}) pixels and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Roll a panoptic image by specified number of pixels"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--roll-x",
        type=int,
        default=0,
        help="Number of pixels to roll horizontally (default: 0)"
    )
    parser.add_argument(
        "--roll-y",
        type=int,
        default=0,
        help="Number of pixels to roll vertically (default: 0)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (default: adds '_rolled' to input filename)"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input image not found: {args.input_path}")
    
    # Generate output path if not provided
    if args.output is None:
        base, ext = os.path.splitext(args.input_path)
        args.output = f"{base}_rolled{ext}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Roll the image
    roll_image(args.input_path, args.output, args.roll_x, args.roll_y)


if __name__ == "__main__":
    main()

