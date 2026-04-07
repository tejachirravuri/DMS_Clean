#!/bin/bash
# Upload experiment results from remote to Google Drive, then download to local
# Usage: bash upload_results.sh [method]
#   method: rclone (default), tar+scp, gdown

set -e
METHOD="${1:-rclone}"
RESULTS_DIR="$(dirname "$0")/results"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: No results directory found at $RESULTS_DIR"
    exit 1
fi

echo "=== Results summary ==="
du -sh "$RESULTS_DIR"
find "$RESULTS_DIR" -name "*.json" | wc -l
echo "JSON files"
find "$RESULTS_DIR" -name "*.png" | wc -l
echo "PNG figures"

if [ "$METHOD" = "rclone" ]; then
    echo "=== Uploading results with rclone ==="
    # ---- EDIT THIS ----
    GDRIVE_DEST="gdrive:DMS_Experiment_Results"
    # --------------------
    rclone copy "$RESULTS_DIR" "$GDRIVE_DEST/" --progress
    echo "Uploaded to: $GDRIVE_DEST"

elif [ "$METHOD" = "tar" ]; then
    echo "=== Creating tar archive ==="
    ARCHIVE="dms_results_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$ARCHIVE" -C "$(dirname "$RESULTS_DIR")" results/
    echo "Archive: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
    echo ""
    echo "Download to local machine:"
    echo "  scp gchi@134.109.184.66:~/DMS_Clean/$ARCHIVE G:/Teja_Master_Thesis/DMS_Clean/"
    echo ""
    echo "Then from local, upload to Google Drive manually or:"
    echo "  rclone copy $ARCHIVE gdrive:DMS_Experiment_Results/"

elif [ "$METHOD" = "scp" ]; then
    echo "=== Download via SCP ==="
    echo "Run FROM your local machine:"
    echo ""
    echo "  scp -r gchi@134.109.184.66:~/DMS_Clean/results/ G:/Teja_Master_Thesis/DMS_Clean/results/"
fi

echo ""
echo "=== Done ==="
