#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints
echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
$CMD $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }

echo "Downloading sam2.1_hiera_small.pt checkpoint..."
$CMD $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }

echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
$CMD $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "All SAM2 checkpoints are downloaded successfully."

####################

mkdir sapiens
cd sapiens
mkdir torchscript
cd torchscript
mkdir pose
cd pose
mkdir checkpoints
cd checkpoints
echo "Downloading sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 checkpoint..."
$CMD https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"; exit 1; }
echo "Downloading sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2 checkpoint..."
$CMD https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"; exit 1; }
echo "Downloading sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 checkpoint..."
$CMD https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"; exit 1; }
cd ../../../
mkdir sapiens
cd sapiens
mkdir bfloat16
cd bfloat16
mkdir pose
cd pose
mkdir checkpoints
cd checkpoints
echo "Downloading sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 checkpoint..."
$CMD https://huggingface.co/facebook/sapiens-pose-0.3b-bfloat16/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.3b-bfloat16/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2"; exit 1; }
echo "Downloading sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2 checkpoint..."
$CMD https://huggingface.co/facebook/sapiens-pose-0.6b-bfloat16/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.6b-bfloat16/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2"; exit 1; }
echo "Downloading sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 checkpoint..."
$CMD https://huggingface.co/facebook/sapiens-pose-1b-bfloat16/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-1b-bfloat16/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2"; exit 1; }
cd ../../../

echo "All Sapiens checkpoints are downloaded successfully."

####################