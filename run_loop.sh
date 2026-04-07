#!/bin/bash
# =============================================================================
# DMS-Raptor: Automated Video-by-Video Experiment Loop
# =============================================================================
# Downloads one video at a time from Google Drive, runs all experiments on CPU,
# uploads results to Google Drive, cleans up, repeats.
#
# Usage:
#   bash run_loop.sh                    # run all videos
#   bash run_loop.sh --dry-run          # just show what would be done
#   bash run_loop.sh --video glass_ins  # run only one specific video
#
# Prerequisites:
#   - rclone configured with "gdrive:" remote
#   - Models already in ~/DMS_Clean/models/
#   - Videos on Google Drive at the paths defined below
# =============================================================================

set -e

# ---- CONFIGURATION (EDIT THESE) ----
GDRIVE_VIDEOS="gdrive:DMS_Test_videos"
GDRIVE_RESULTS="gdrive:DMS_Experiment_Results"
WORK_DIR="$HOME/DMS_Clean"
MODELS_DIR="$WORK_DIR/models"
VIDEOS_DIR="$WORK_DIR/videos"
RESULTS_DIR="$WORK_DIR/results"
DEVICE="cpu"
# MAX_FRAMES=0 means all frames; set to e.g. 500 for quick test
MAX_FRAMES=0
ANNOTATE_POLICIES="conf_ema"
# ------------------------------------

# Find Python: try venv, then python3, then python
if [ -f "$HOME/DMS_Raptor/venv/bin/python" ]; then
    PYTHON="$HOME/DMS_Raptor/venv/bin/python"
elif [ -f "$WORK_DIR/venv/bin/python" ]; then
    PYTHON="$WORK_DIR/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: No python found"
    exit 1
fi

# Video list: "gdrive_subpath|local_material_dir|short_name"
# Edit this list to match your Google Drive structure
VIDEOS=(
    "glass/glass_ins.mp4|glass|glass_ins"
    "glass/021_YUN_0001_111.mp4|glass|021_YUN_0001_111"
    "glass/161_YUN_0001_96.mp4|glass|161_YUN_0001_96"
    "glass/20190911-174-01.mp4|glass|174-01"
    "glass/20190912-233.mp4|glass|233"
    "glass/20190916-633-01.mp4|glass|633-01"
    "glass/20190916-722.mp4|glass|722"
    "glass/20190918-515.mp4|glass|515"
    "glass/YUN_0001_58.mp4|glass|YUN_0001_58"
    "porcelain/UAV_porcelain.mp4|porcelain|UAV_porcelain"
    "porcelain/porce.mp4|porcelain|porce"
    "porcelain/porcelain_maybe.mp4|porcelain|porcelain_maybe"
)

DRY_RUN=false
ONLY_VIDEO=""

# Parse args
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --video) shift; ONLY_VIDEO="$2" ;;
        --video=*) ONLY_VIDEO="${arg#*=}" ;;
        --max-frames=*) MAX_FRAMES="${arg#*=}" ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERR]${NC} $1"; }

# Check prerequisites
check_prereqs() {
    if ! command -v rclone &>/dev/null; then
        err "rclone not found. Install it first."
        exit 1
    fi
    if ! rclone listremotes | grep -q "gdrive:"; then
        err "rclone 'gdrive:' remote not configured. Run: rclone config"
        exit 1
    fi
    if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls $MODELS_DIR/*.pt 2>/dev/null)" ]; then
        err "No models found in $MODELS_DIR"
        exit 1
    fi
    log "Disk free: $(df -h ~ | tail -1 | awk '{print $4}')"
    log "Python: $PYTHON ($($PYTHON --version 2>&1))"
}

# Download one video from Google Drive
download_video() {
    local gdrive_path="$1"
    local material="$2"
    local dest_dir="$VIDEOS_DIR/$material"

    mkdir -p "$dest_dir"
    log "Downloading: $gdrive_path"

    if $DRY_RUN; then
        log "[DRY-RUN] Would download $GDRIVE_VIDEOS/$gdrive_path -> $dest_dir/"
        return 0
    fi

    rclone copy "$GDRIVE_VIDEOS/$gdrive_path" "$dest_dir/" --progress
    ok "Downloaded to $dest_dir/"
    ls -lh "$dest_dir/"
}

# Run all experiments on one video
run_experiments() {
    local material="$1"
    local short_name="$2"

    log "Running experiments: $material/$short_name (device=$DEVICE)"

    if $DRY_RUN; then
        log "[DRY-RUN] Would run: python dms_experiment.py run-all --device $DEVICE --max-frames $MAX_FRAMES"
        return 0
    fi

    cd "$WORK_DIR"

    # Build max-frames arg
    MF_ARG=""
    if [ "$MAX_FRAMES" -gt 0 ] 2>/dev/null; then
        MF_ARG="--max-frames $MAX_FRAMES"
    fi

    $PYTHON dms_experiment.py run-all \
        --models-dir "$MODELS_DIR" \
        --videos-dir "$VIDEOS_DIR" \
        --results-dir "$RESULTS_DIR" \
        --device "$DEVICE" \
        --annotate-policies $ANNOTATE_POLICIES \
        $MF_ARG

    ok "Experiments complete for $material/$short_name"
}

# Upload results to Google Drive
upload_results() {
    local short_name="$1"

    log "Uploading results to Google Drive..."

    if $DRY_RUN; then
        log "[DRY-RUN] Would upload $RESULTS_DIR/ -> $GDRIVE_RESULTS/"
        return 0
    fi

    rclone copy "$RESULTS_DIR/" "$GDRIVE_RESULTS/" --progress
    ok "Results uploaded to $GDRIVE_RESULTS/"
}

# Clean up video and results to free space
cleanup() {
    local material="$1"
    local short_name="$2"

    log "Cleaning up to free space..."

    if $DRY_RUN; then
        log "[DRY-RUN] Would delete $VIDEOS_DIR/ and $RESULTS_DIR/"
        return 0
    fi

    # Remove downloaded video
    rm -rf "$VIDEOS_DIR"
    # Remove local results (already uploaded to Drive)
    rm -rf "$RESULTS_DIR"

    ok "Cleaned up. Disk free: $(df -h ~ | tail -1 | awk '{print $4}')"
}

# =============================================================================
# MAIN LOOP
# =============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "  DMS-Raptor Automated Experiment Loop"
    echo "  Device: $DEVICE | Videos: ${#VIDEOS[@]}"
    echo "  Max frames: $([ "$MAX_FRAMES" -gt 0 ] 2>/dev/null && echo $MAX_FRAMES || echo 'ALL')"
    echo "============================================================"
    echo ""

    check_prereqs

    TOTAL=${#VIDEOS[@]}
    DONE=0
    FAILED=0

    for entry in "${VIDEOS[@]}"; do
        IFS='|' read -r gdrive_path material short_name <<< "$entry"

        # Skip if --video filter is set and doesn't match
        if [ -n "$ONLY_VIDEO" ] && [ "$short_name" != "$ONLY_VIDEO" ]; then
            continue
        fi

        DONE=$((DONE + 1))
        echo ""
        echo "============================================================"
        log "VIDEO $DONE/$TOTAL: $material/$short_name"
        echo "============================================================"

        # Step 1: Download
        if ! download_video "$gdrive_path" "$material"; then
            err "Failed to download $short_name. Skipping."
            FAILED=$((FAILED + 1))
            continue
        fi

        # Step 2: Run experiments
        if ! run_experiments "$material" "$short_name"; then
            err "Experiments failed for $short_name."
            FAILED=$((FAILED + 1))
            # Still try to upload partial results
        fi

        # Step 3: Upload results
        if ! upload_results "$short_name"; then
            err "Upload failed for $short_name. Results kept locally."
            warn "Manual upload needed: rclone copy $RESULTS_DIR/ $GDRIVE_RESULTS/"
            continue
        fi

        # Step 4: Cleanup
        cleanup "$material" "$short_name"

        ok "Completed $material/$short_name ($DONE/$TOTAL)"
    done

    echo ""
    echo "============================================================"
    echo "  LOOP COMPLETE"
    echo "  Processed: $DONE | Failed: $FAILED"
    echo "  Results on Google Drive: $GDRIVE_RESULTS"
    echo "============================================================"
}

main "$@"
