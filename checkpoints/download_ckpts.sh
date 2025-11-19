#!/bin/bash

# Default to mini if no arguments provided
CHECKPOINTS="mini"

# Set current dir
CURRENT_DIR=$(pwd)

# Parse command line arguments
while getopts "c:" opt; do
    case $opt in
        c)
            CHECKPOINTS="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-c checkpoints]"
            echo "  -c: Comma-separated list of checkpoint shortnames or special keywords"
            echo "      Special keywords: all, mini, none"
            echo "      Available checkpoints: sam21t, sam21s, sam21b, sam21l, sapiens03ts, sapiens03b16, sapiens06ts, sapiens06b16, sapiens1bts, sapiens1b16"
            exit 1
            ;;
    esac
done

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define checkpoint mappings
declare -A CHECKPOINT_MAP
CHECKPOINT_MAP["sam21t"]="sam2.1_hiera_tiny.pt"
CHECKPOINT_MAP["sam21s"]="sam2.1_hiera_small.pt"
CHECKPOINT_MAP["sam21b"]="sam2.1_hiera_base_plus.pt"
CHECKPOINT_MAP["sam21l"]="sam2.1_hiera_large.pt"
CHECKPOINT_MAP["sapiens03ts"]="sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
CHECKPOINT_MAP["sapiens03b16"]="sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2"
CHECKPOINT_MAP["sapiens06ts"]="sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"
CHECKPOINT_MAP["sapiens06b16"]="sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2"
CHECKPOINT_MAP["sapiens1bts"]="sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
CHECKPOINT_MAP["sapiens1b16"]="sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2"

# Define special keyword mappings
case "$CHECKPOINTS" in
    "all")
        SELECTED_CHECKPOINTS="sam21t,sam21s,sam21b,sam21l,sapiens03ts,sapiens03b16,sapiens06ts,sapiens06b16,sapiens1bts,sapiens1b16"
        ;;
    "mini")
        SELECTED_CHECKPOINTS="sam21t,sam21b,sapiens03ts,sapiens03b16"
        ;;
    "none")
        SELECTED_CHECKPOINTS=""
        ;;
    *)
        SELECTED_CHECKPOINTS="$CHECKPOINTS"
        ;;
esac

echo "Selected checkpoints: $SELECTED_CHECKPOINTS"

# Function to check if a checkpoint should be downloaded
should_download() {
    local checkpoint="$1"
    if [[ "$SELECTED_CHECKPOINTS" == "" ]]; then
        return 1
    fi
    IFS=',' read -ra CHECKPOINT_ARRAY <<< "$SELECTED_CHECKPOINTS"
    for selected in "${CHECKPOINT_ARRAY[@]}"; do
        if [[ "$selected" == "$checkpoint" ]]; then
            return 0
        fi
    done
    return 1
}

#################### SAM 2.1 ####################

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints
if should_download "sam21t"; then
    if [ -f "sam2.1_hiera_tiny.pt" ]; then
        echo "sam2.1_hiera_tiny.pt already exists, skipping download."
    else
        echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
        $CMD $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }
    fi
fi

if should_download "sam21s"; then
    if [ -f "sam2.1_hiera_small.pt" ]; then
        echo "sam2.1_hiera_small.pt already exists, skipping download."
    else
        echo "Downloading sam2.1_hiera_small.pt checkpoint..."
        $CMD $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }
    fi
fi

if should_download "sam21b"; then
    if [ -f "sam2.1_hiera_base_plus.pt" ]; then
        echo "sam2.1_hiera_base_plus.pt already exists, skipping download."
    else
        echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
        $CMD $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; exit 1; }
    fi
fi

if should_download "sam21l"; then
    if [ -f "sam2.1_hiera_large.pt" ]; then
        echo "sam2.1_hiera_large.pt already exists, skipping download."
    else
        echo "Downloading sam2.1_hiera_large.pt checkpoint..."
        $CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }
    fi
fi

echo "SAM2 checkpoints download completed."


#################### Sapiens ####################

# Check if any Sapiens checkpoints need to be downloaded
SAPIENS_NEEDED=false
if should_download "sapiens03ts" || should_download "sapiens06ts" || should_download "sapiens1bts"; then
    SAPIENS_NEEDED=true
fi

if [ "$SAPIENS_NEEDED" = true ]; then
    mkdir -p sapiens/torchscript/pose/checkpoints
    cd sapiens/torchscript/pose/checkpoints
    
    if should_download "sapiens03ts"; then
        if [ -f "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2" ]; then
            echo "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 already exists, skipping download."
        else
            echo "Downloading sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 checkpoint..."
            $CMD https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"; exit 1; }
        fi
    fi
    
    if should_download "sapiens06ts"; then
        if [ -f "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2" ]; then
            echo "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2 already exists, skipping download."
        else
            echo "Downloading sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2 checkpoint..."
            $CMD https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"; exit 1; }
        fi
    fi
    
    if should_download "sapiens1bts"; then
        if [ -f "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2" ]; then
            echo "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 already exists, skipping download."
        else
            echo "Downloading sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 checkpoint..."
            $CMD https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"; exit 1; }
        fi
    fi
    
    cd $CURRENT_DIR
fi

# Check if any Sapiens bfloat16 checkpoints need to be downloaded
SAPIENS_BFLOAT16_NEEDED=false
if should_download "sapiens03b16" || should_download "sapiens06b16" || should_download "sapiens1b16"; then
    SAPIENS_BFLOAT16_NEEDED=true
fi

if [ "$SAPIENS_BFLOAT16_NEEDED" = true ]; then
    mkdir -p sapiens/bfloat16/pose/checkpoints
    cd sapiens/bfloat16/pose/checkpoints
    
    if should_download "sapiens03b16"; then
        if [ -f "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2" ]; then
            echo "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2 already exists, skipping download."
        else
            echo "Downloading sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2 checkpoint..."
            $CMD https://huggingface.co/facebook/sapiens-pose-0.3b-bfloat16/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.3b-bfloat16/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2"; exit 1; }
        fi
    fi
    
    if should_download "sapiens06b16"; then
        if [ -f "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2" ]; then
            echo "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2 already exists, skipping download."
        else
            echo "Downloading sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2 checkpoint..."
            $CMD https://huggingface.co/facebook/sapiens-pose-0.6b-bfloat16/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-0.6b-bfloat16/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2"; exit 1; }
        fi
    fi
    
    if should_download "sapiens1b16"; then
        if [ -f "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2" ]; then
            echo "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2 already exists, skipping download."
        else
            echo "Downloading sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2 checkpoint..."
            $CMD https://huggingface.co/facebook/sapiens-pose-1b-bfloat16/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2 || { echo "Failed to download checkpoint from https://huggingface.co/facebook/sapiens-pose-1b-bfloat16/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2"; exit 1; }
        fi
    fi
    
    cd ../../../
fi

echo "Sapiens checkpoints download completed."

####################