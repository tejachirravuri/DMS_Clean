#!/bin/bash
# =============================================================================
# DMS-Raptor: Automated Batch Experiment Loop
# =============================================================================
# Strategy:
#   - Detection quality + Frame validation → GPU (same results, 100x faster)
#   - Overnight pipeline (benchmarking)    → CPU (accurate timing traces)
#   - Timing trials                        → SKIPPED (overnight gives per-frame timing)
#
# Batches 4 videos at a time, uploads to Google Drive between batches.
#
# Usage:
#   bash run_loop.sh                    # all 12 videos in 3 batches
#   bash run_loop.sh --batch 1          # only batch 1 (videos 1-4)
#   bash run_loop.sh --batch 2          # only batch 2 (videos 5-8)
#   bash run_loop.sh --batch 3          # only batch 3 (videos 9-12)
#   bash run_loop.sh --dry-run          # preview without running
#   bash run_loop.sh --max-frames=500   # quick test
# =============================================================================

set -e

# ---- CONFIGURATION ----
GDRIVE_VIDEOS="gdrive:DMS_Test_videos"
GDRIVE_RESULTS="gdrive:DMS_Experiment_Results"
WORK_DIR="$HOME/DMS_Clean"
MODELS_DIR="$WORK_DIR/models"
VIDEOS_DIR="$WORK_DIR/videos"
RESULTS_DIR="$WORK_DIR/results"
MAX_FRAMES=0
ANNOTATE_POLICIES="conf_ema"
BATCH_SIZE=4
# ------------------------

# Find Python
if [ -f "$HOME/DMS_Raptor/venv/bin/python" ]; then
    PYTHON="$HOME/DMS_Raptor/venv/bin/python"
elif [ -f "$WORK_DIR/venv/bin/python" ]; then
    PYTHON="$WORK_DIR/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: No python found"; exit 1
fi

# All 12 videos: "gdrive_subpath|material|short_name"
ALL_VIDEOS=(
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

# Parse args
DRY_RUN=false
ONLY_BATCH=0  # 0 = all batches
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --batch) shift ;;
        --batch=*) ONLY_BATCH="${arg#*=}" ;;
        [1-3]) ONLY_BATCH="$arg" ;;
        --max-frames=*) MAX_FRAMES="${arg#*=}" ;;
    esac
done

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERR]${NC} $1"; }

check_prereqs() {
    command -v rclone &>/dev/null || { err "rclone not found"; exit 1; }
    rclone listremotes | grep -q "gdrive:" || { err "gdrive: not configured"; exit 1; }
    [ -d "$MODELS_DIR" ] && ls $MODELS_DIR/*.pt &>/dev/null || { err "No models in $MODELS_DIR"; exit 1; }
    log "Python: $PYTHON ($($PYTHON --version 2>&1))"
    log "Disk free: $(df -h ~ | tail -1 | awk '{print $4}')"

    # Check GPU availability
    if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        HAS_GPU=true
        log "GPU: available (will use for detection quality + validation)"
    else
        HAS_GPU=false
        warn "No GPU detected. All phases will use CPU (slower)."
    fi
}

# Download one video
download_video() {
    local gdrive_path="$1" material="$2"
    mkdir -p "$VIDEOS_DIR/$material"
    log "Downloading: $gdrive_path"
    $DRY_RUN && { log "[DRY-RUN] skip download"; return 0; }
    rclone copy "$GDRIVE_VIDEOS/$gdrive_path" "$VIDEOS_DIR/$material/" --progress
    ok "Downloaded"
}

# Run experiments for one video: GPU phases first, then CPU benchmarking
run_single_video() {
    local material="$1" short_name="$2"
    cd "$WORK_DIR"

    MF_ARG=""
    [ "$MAX_FRAMES" -gt 0 ] 2>/dev/null && MF_ARG="--max-frames $MAX_FRAMES"

    # ---- Phase 1: Detection Quality on GPU (both models per frame → P/R/F1) ----
    if $HAS_GPU; then
        GPU_DEV="cuda"
    else
        GPU_DEV="cpu"
    fi

    log "Phase 1/4: Detection quality ($GPU_DEV)..."
    $DRY_RUN || $PYTHON dms_experiment.py detection \
        --models-dir "$MODELS_DIR" --videos-dir "$VIDEOS_DIR" \
        --results-dir "$RESULTS_DIR" --device "$GPU_DEV" $MF_ARG

    # ---- Phase 2: Frame-level validation on GPU (IoU + 30 sample images) ----
    log "Phase 2/4: Frame validation ($GPU_DEV)..."
    $DRY_RUN || $PYTHON dms_experiment.py validation \
        --models-dir "$MODELS_DIR" --videos-dir "$VIDEOS_DIR" \
        --results-dir "$RESULTS_DIR" --device "$GPU_DEV" --samples 30 $MF_ARG

    # ---- Phase 3: Overnight pipeline on CPU (benchmarkable timing) ----
    log "Phase 3/4: Overnight pipeline (cpu - benchmarking)..."
    $DRY_RUN || $PYTHON dms_experiment.py overnight \
        --models-dir "$MODELS_DIR" --videos-dir "$VIDEOS_DIR" \
        --results-dir "$RESULTS_DIR" --device cpu \
        --annotate-policies $ANNOTATE_POLICIES $MF_ARG

    # ---- Phase 4: Plots + Report ----
    log "Phase 4/4: Generating plots + report..."
    $DRY_RUN || $PYTHON dms_experiment.py plots --results-dir "$RESULTS_DIR"
    $DRY_RUN || $PYTHON dms_experiment.py report --results-dir "$RESULTS_DIR"

    ok "All phases complete for $material/$short_name"
}

# Upload results to Google Drive
upload_results() {
    log "Uploading results to Google Drive..."
    $DRY_RUN && { log "[DRY-RUN] skip upload"; return 0; }
    rclone copy "$RESULTS_DIR/" "$GDRIVE_RESULTS/" --progress
    ok "Results uploaded to $GDRIVE_RESULTS/"
}

# Clean up videos and results
cleanup() {
    log "Cleaning up..."
    $DRY_RUN && { log "[DRY-RUN] skip cleanup"; return 0; }
    rm -rf "$VIDEOS_DIR" "$RESULTS_DIR"
    ok "Cleaned. Disk free: $(df -h ~ | tail -1 | awk '{print $4}')"
}

# =============================================================================
# MAIN: Process in batches of 4
# =============================================================================
main() {
    echo ""
    echo "============================================================"
    echo "  DMS-Raptor: Batch Experiment Pipeline"
    echo "  Total videos: ${#ALL_VIDEOS[@]} | Batch size: $BATCH_SIZE"
    echo "  GPU phases: detection quality, frame validation"
    echo "  CPU phases: overnight pipeline (benchmarking)"
    echo "  Timing trials: SKIPPED (overnight has per-frame traces)"
    echo "  Max frames: $([ "$MAX_FRAMES" -gt 0 ] 2>/dev/null && echo $MAX_FRAMES || echo 'ALL')"
    echo "============================================================"
    echo ""

    HAS_GPU=false
    check_prereqs

    TOTAL=${#ALL_VIDEOS[@]}
    NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
    GLOBAL_DONE=0
    GLOBAL_FAILED=0

    for ((batch=1; batch<=NUM_BATCHES; batch++)); do
        # Skip batches if --batch filter is set
        if [ "$ONLY_BATCH" -gt 0 ] 2>/dev/null && [ "$batch" -ne "$ONLY_BATCH" ]; then
            continue
        fi

        START_IDX=$(( (batch - 1) * BATCH_SIZE ))
        END_IDX=$(( START_IDX + BATCH_SIZE ))
        [ "$END_IDX" -gt "$TOTAL" ] && END_IDX=$TOTAL

        echo ""
        echo "============================================================"
        log "BATCH $batch/$NUM_BATCHES (videos $((START_IDX+1))-$END_IDX of $TOTAL)"
        echo "============================================================"

        # Process each video in the batch: download → run → upload → delete → next
        for ((i=START_IDX; i<END_IDX; i++)); do
            IFS='|' read -r gdrive_path material short_name <<< "${ALL_VIDEOS[$i]}"
            GLOBAL_DONE=$((GLOBAL_DONE + 1))

            echo ""
            log "--- Video $GLOBAL_DONE/$TOTAL: $material/$short_name ---"

            # Download
            if ! download_video "$gdrive_path" "$material"; then
                err "Download failed: $short_name. Skipping."
                GLOBAL_FAILED=$((GLOBAL_FAILED + 1))
                continue
            fi

            # Run all phases
            if ! run_single_video "$material" "$short_name"; then
                err "Experiments failed: $short_name."
                GLOBAL_FAILED=$((GLOBAL_FAILED + 1))
            fi

            # Upload THIS video's results immediately
            log "Uploading results for $short_name..."
            if ! upload_results; then
                err "Upload failed for $short_name!"
                warn "Results still in: $RESULTS_DIR"
            fi

            # Delete video AND results to free space for next video
            cleanup "$material" "$short_name"

            ok "Done: $material/$short_name ($GLOBAL_DONE/$TOTAL)"
        done

        echo ""
        ok "BATCH $batch/$NUM_BATCHES COMPLETE"
        echo ""
        if [ "$batch" -lt "$NUM_BATCHES" ] && [ "$ONLY_BATCH" -eq 0 ] 2>/dev/null; then
            log "Starting next batch in 10 seconds... (Ctrl+C to pause)"
            $DRY_RUN || sleep 10
        fi
    done

    echo ""
    echo "============================================================"
    echo "  ALL DONE"
    echo "  Processed: $GLOBAL_DONE | Failed: $GLOBAL_FAILED"
    echo "  Results: $GDRIVE_RESULTS"
    echo "============================================================"
}

main "$@"
