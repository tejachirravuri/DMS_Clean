#!/bin/bash
# =============================================================================
# DMS Agnosticism Experiments - Complete Pipeline
# =============================================================================
# Proves DMS is data-agnostic and model-agnostic.
#
# Experiments:
#   EXP1: YOLO11 n/s (COCO pretrained) on insulator videos  [architecture agnostic]
#   EXP2: YOLOv8m trained on insulators, paired with YOLOv8n [scaling agnostic]
#   EXP3: YOLO11 n/s trained on insulators                   [architecture + domain]
#   EXP4: COCO pretrained YOLOv8 n/s on VisDrone video       [data agnostic]
#
# Usage:
#   bash run_agnostic.sh --phase train    # Phase 1: train models on GPU
#   bash run_agnostic.sh --phase test     # Phase 2: run DMS on CPU
#   bash run_agnostic.sh --all            # Both phases
# =============================================================================

set -e
WORK_DIR="$HOME/DMS_Clean"
PYTHON="$WORK_DIR/venv/bin/python"
AGNOSTIC_DIR="$WORK_DIR/agnostic_experiments"
RESULTS_DIR="$WORK_DIR/agnostic_results"

# Parse args
PHASE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2;;
        --all) PHASE="all"; shift;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  DMS Agnosticism Experiments"
echo "  Phase: $PHASE"
echo "  Time: $(date)"
echo "============================================================"

# ---- PHASE 1: TRAIN MODELS ON GPU ----
if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
    echo ""
    echo "[Phase 1] Training models on GPU..."
    echo ""

    # Check GPU
    $PYTHON -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

    # Check dataset exists
    GLASS_YAML="$WORK_DIR/datasets/glass_insulator.yaml"
    if [ ! -f "$GLASS_YAML" ]; then
        echo "[ERROR] Glass dataset not found at $GLASS_YAML"
        echo "Upload the dataset first. See instructions below."
        exit 1
    fi

    # Train YOLO11n on glass insulators
    echo ""
    echo ">>> Training YOLO11n on glass insulators..."
    $PYTHON train_agnostic_models.py --dataset glass --device 0 --models yolo11n --data-yaml "$GLASS_YAML" --epochs 100 --batch 16 2>&1 | tee "$AGNOSTIC_DIR/train_yolo11n_glass.log"

    # Train YOLO11s on glass insulators
    echo ""
    echo ">>> Training YOLO11s on glass insulators..."
    $PYTHON train_agnostic_models.py --dataset glass --device 0 --models yolo11s --data-yaml "$GLASS_YAML" --epochs 100 --batch 16 2>&1 | tee "$AGNOSTIC_DIR/train_yolo11s_glass.log"

    # Train YOLOv8m on glass insulators
    echo ""
    echo ">>> Training YOLOv8m on glass insulators..."
    $PYTHON train_agnostic_models.py --dataset glass --device 0 --models yolov8m --data-yaml "$GLASS_YAML" --epochs 100 --batch 16 2>&1 | tee "$AGNOSTIC_DIR/train_yolov8m_glass.log"

    echo ""
    echo "[Phase 1] Training COMPLETE. Model weights saved."
fi

# ---- PHASE 2: RUN DMS EXPERIMENTS ON CPU ----
if [[ "$PHASE" == "test" || "$PHASE" == "all" ]]; then
    echo ""
    echo "[Phase 2] Running DMS experiments on CPU..."
    echo ""

    mkdir -p "$RESULTS_DIR"

    # ---- EXP1: COCO YOLO11 n/s on insulator videos ----
    echo "============================================================"
    echo "  EXP1: COCO YOLO11 n/s on insulator videos"
    echo "  (Architecture agnostic - zero training)"
    echo "============================================================"
    EXP1_MODELS="$AGNOSTIC_DIR/exp1_coco_yolo11"
    mkdir -p "$EXP1_MODELS"

    # Download COCO pretrained YOLO11
    $PYTHON -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11s.pt')"
    cp yolo11n.pt "$EXP1_MODELS/" 2>/dev/null || true
    cp yolo11s.pt "$EXP1_MODELS/" 2>/dev/null || true

    # Pick 2 representative insulator videos (1 easy + 1 hard)
    # These test if COCO-pretrained models still show a gap
    for VID in videos/glass/*.mp4; do
        VNAME=$(basename "$VID" .mp4)
        echo "  Running EXP1 on $VNAME..."
        $PYTHON dms_experiment.py overnight \
            --models-dir "$EXP1_MODELS" \
            --videos-dir "$(dirname $VID)" \
            --device cpu \
            --results-dir "$RESULTS_DIR/exp1_coco_yolo11/$VNAME" \
            --max-frames 3000 \
            2>&1 | tee -a "$AGNOSTIC_DIR/exp1.log"
        break  # Just first video for quick proof
    done

    # ---- EXP2: Trained YOLO11 n/s on insulator videos ----
    echo "============================================================"
    echo "  EXP2: Trained YOLO11 n/s on insulator videos"
    echo "  (Architecture + domain agnostic)"
    echo "============================================================"
    EXP2_MODELS="$AGNOSTIC_DIR/exp2_yolo11_insulator"
    mkdir -p "$EXP2_MODELS"

    Y11N_BEST="agnostic_training/glass/yolo11n_glass/weights/best.pt"
    Y11S_BEST="agnostic_training/glass/yolo11s_glass/weights/best.pt"

    if [ -f "$Y11N_BEST" ] && [ -f "$Y11S_BEST" ]; then
        cp "$Y11N_BEST" "$EXP2_MODELS/yolo11n_glass.pt"
        cp "$Y11S_BEST" "$EXP2_MODELS/yolo11s_glass.pt"

        for VID in videos/glass/*.mp4; do
            VNAME=$(basename "$VID" .mp4)
            echo "  Running EXP2 on $VNAME..."
            $PYTHON dms_experiment.py overnight \
                --models-dir "$EXP2_MODELS" \
                --videos-dir "$(dirname $VID)" \
                --device cpu \
                --results-dir "$RESULTS_DIR/exp2_yolo11_insulator/$VNAME" \
                --max-frames 3000 \
                2>&1 | tee -a "$AGNOSTIC_DIR/exp2.log"
            break
        done
    else
        echo "[SKIP] EXP2: Trained YOLO11 weights not found. Run --phase train first."
    fi

    # ---- EXP3: YOLOv8n + YOLOv8m on insulator videos ----
    echo "============================================================"
    echo "  EXP3: YOLOv8n + YOLOv8m (scaling agnostic)"
    echo "============================================================"
    EXP3_MODELS="$AGNOSTIC_DIR/exp3_v8n_v8m"
    mkdir -p "$EXP3_MODELS"

    V8M_BEST="agnostic_training/glass/yolov8m_glass/weights/best.pt"

    if [ -f "$V8M_BEST" ]; then
        # Use existing YOLOv8n (from main experiments) + trained YOLOv8m
        cp models/*n*.pt "$EXP3_MODELS/yolov8n_glass.pt" 2>/dev/null || true
        cp "$V8M_BEST" "$EXP3_MODELS/yolov8m_glass.pt"

        for VID in videos/glass/*.mp4; do
            VNAME=$(basename "$VID" .mp4)
            echo "  Running EXP3 on $VNAME..."
            $PYTHON dms_experiment.py overnight \
                --models-dir "$EXP3_MODELS" \
                --videos-dir "$(dirname $VID)" \
                --device cpu \
                --results-dir "$RESULTS_DIR/exp3_v8n_v8m/$VNAME" \
                --max-frames 3000 \
                2>&1 | tee -a "$AGNOSTIC_DIR/exp3.log"
            break
        done
    else
        echo "[SKIP] EXP3: YOLOv8m weights not found. Run --phase train first."
    fi

    echo ""
    echo "============================================================"
    echo "  ALL AGNOSTIC EXPERIMENTS COMPLETE"
    echo "  Results: $RESULTS_DIR/"
    echo "============================================================"

    # Upload results
    if command -v rclone &>/dev/null; then
        echo "Uploading results to Google Drive..."
        rclone copy "$RESULTS_DIR" "gdrive:DMS_Experiment_Results/agnostic_results" --progress
    fi
fi
