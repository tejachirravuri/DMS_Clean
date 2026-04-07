#!/bin/bash
# Download videos from Google Drive to remote machine
# Usage: bash download_videos.sh
#
# OPTION 1: gdown (pip install gdown)
# Replace FOLDER_ID with your Google Drive folder ID
# The folder ID is the last part of the shared folder URL:
# https://drive.google.com/drive/folders/FOLDER_ID
#
# OPTION 2: rclone (more reliable for large files)
# First configure: rclone config  (add Google Drive remote named "gdrive")

set -e

METHOD="${1:-gdown}"  # gdown or rclone

VIDEOS_DIR="$(dirname "$0")/videos"
MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$VIDEOS_DIR/glass_insulator_videos" "$VIDEOS_DIR/porcelain_insulator_videos" "$MODELS_DIR"

if [ "$METHOD" = "gdown" ]; then
    pip install -q gdown

    echo "=== Downloading videos with gdown ==="
    echo "Set your Google Drive folder IDs below:"

    # ---- EDIT THESE ----
    GLASS_FOLDER_ID="YOUR_GLASS_VIDEOS_FOLDER_ID"
    PORCELAIN_FOLDER_ID="YOUR_PORCELAIN_VIDEOS_FOLDER_ID"
    MODELS_FOLDER_ID="YOUR_MODELS_FOLDER_ID"
    # --------------------

    if [ "$GLASS_FOLDER_ID" != "YOUR_GLASS_VIDEOS_FOLDER_ID" ]; then
        gdown --folder "https://drive.google.com/drive/folders/$GLASS_FOLDER_ID" -O "$VIDEOS_DIR/glass_insulator_videos/"
        gdown --folder "https://drive.google.com/drive/folders/$PORCELAIN_FOLDER_ID" -O "$VIDEOS_DIR/porcelain_insulator_videos/"
        gdown --folder "https://drive.google.com/drive/folders/$MODELS_FOLDER_ID" -O "$MODELS_DIR/"
    else
        echo "ERROR: Edit this script and set your Google Drive folder IDs!"
        echo "  GLASS_FOLDER_ID, PORCELAIN_FOLDER_ID, MODELS_FOLDER_ID"
        exit 1
    fi

elif [ "$METHOD" = "rclone" ]; then
    echo "=== Downloading videos with rclone ==="
    echo "Make sure 'rclone config' has a remote named 'gdrive'"

    # ---- EDIT THESE (Google Drive paths) ----
    GDRIVE_GLASS="gdrive:MT_Chirravuri/input_videos/glass_insulator_videos"
    GDRIVE_PORCELAIN="gdrive:MT_Chirravuri/input_videos/porcelain_insulator_videos"
    GDRIVE_MODELS="gdrive:MT_Chirravuri/models"
    # ------------------------------------------

    rclone copy "$GDRIVE_GLASS" "$VIDEOS_DIR/glass_insulator_videos/" --progress
    rclone copy "$GDRIVE_PORCELAIN" "$VIDEOS_DIR/porcelain_insulator_videos/" --progress
    rclone copy "$GDRIVE_MODELS" "$MODELS_DIR/" --progress

elif [ "$METHOD" = "scp" ]; then
    echo "=== Copy from local machine via SCP ==="
    echo "Run this FROM your local machine:"
    echo ""
    echo "  scp -r G:/Teja_Master_Thesis/MT_Chirravuri/input_videos/* gchi@134.109.184.66:~/DMS_Clean/videos/"
    echo "  scp -r G:/Teja_Master_Thesis/MT_Chirravuri/models/*.pt gchi@134.109.184.66:~/DMS_Clean/models/"
    exit 0
fi

echo ""
echo "=== Download complete ==="
ls -la "$VIDEOS_DIR"/glass_insulator_videos/
ls -la "$VIDEOS_DIR"/porcelain_insulator_videos/
ls -la "$MODELS_DIR"/
