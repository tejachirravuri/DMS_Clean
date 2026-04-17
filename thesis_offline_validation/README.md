# Thesis Offline Validation

This folder contains a new, isolated offline scientific validation path for:

Dynamic Model Switching for Real-Time UAV Inspection

Its purpose is to test whether candidate routing features predict when YOLOv8s is more beneficial than YOLOv8n on a frame.

This module is intentionally separate from the existing online DMS runtime, streaming engine, switching controller, timing benchmarks, and plotting pipelines.

## Separation From Online DMS Benchmarking

This module only implements Phase A:

- run both models on the same sampled frames
- build auditable frame-level benefit labels
- compute candidate routing features
- produce a canonical master per-frame table for later statistics and plotting

This module does not implement Phase B:

- no dwell logic
- no switching controller
- no budget guard
- no online latency benchmarking conclusions

## Governance Rules

Every feature and target is registered in `src/feature_registry.py` with:

- `availability`: `pre`, `post_n`, or `offline_only`
- `controller_allowed`: `True` or `False`
- `description`

This prevents target leakage.

Examples:

- `lap_var`: `pre`, controller-allowed
- `n_mean_conf`: `post_n`, controller-allowed
- `extra_dets_s`: `offline_only`, not controller-allowed
- `y_switch_optimal`: `offline_only`, not controller-allowed

## Stage Workflow

1. `01_build_frame_manifest.py`
Builds the canonical frame manifest and optionally extracts sampled frames to disk.

2. `02_run_dual_inference.py`
Runs YOLOv8n and YOLOv8s on all manifest frames and saves detections plus per-frame summaries.

3. `03_build_targets.py`
Matches n/s detections per frame, computes target intermediates, and builds offline labels.

4. `04_extract_features_image.py`
Computes image-only features available before running any detector.

5. `05_extract_features_detection.py`
Computes post-`n` detection-state features from YOLOv8n outputs.

6. `06_extract_features_temporal.py`
Computes temporal deltas, EMAs, confidence-drop features, and zero-detection streaks.

7. `07_extract_features_roi.py`
Computes lightweight ROI features around the highest-confidence n-detection when available.

8. `08_build_master_table.py`
Joins manifest, model summaries, features, intermediates, targets, and governance metadata into one master per-frame table.

## Artifact Layout

Artifacts are written under `outputs/<run_name>/`.

Key files:

- `manifest/frame_manifest.parquet`: one row per sampled frame
- `inference/detections_n.parquet`: YOLOv8n detections
- `inference/detections_s.parquet`: YOLOv8s detections
- `inference/frame_summary_n.parquet`: n-model per-frame summaries
- `inference/frame_summary_s.parquet`: s-model per-frame summaries
- `targets/frame_matching.parquet`: frame-level n/s matching summaries
- `targets/targets.parquet`: target intermediates and final labels
- `features/image_features.parquet`
- `features/detection_features.parquet`
- `features/temporal_features.parquet`
- `features/roi_features.parquet`
- `master/master_table.parquet`: canonical table for later statistical analysis
- `runs/<stage>_run_manifest.json`: per-stage reproducibility manifest
- `snapshots/`: config snapshots used for the run

## Controller-Allowed Inputs

Controller-allowed features are limited to features whose registry entries have:

- `controller_allowed=True`

Use the registry rather than column names alone.

The intended controller-eligible groups are:

- image features (`pre`)
- detection-state features from YOLOv8n (`post_n`)
- temporal features derived from controller-eligible signals
- ROI features derived from YOLOv8n detections, if used intentionally

Offline-only columns must never be used as controller inputs:

- any `s`-only benefit signal
- any target label
- matching/intermediate reference summaries

## Example Commands

Run from repo root:

```bash
python3 thesis_offline_validation/scripts/01_build_frame_manifest.py \
  --config thesis_offline_validation/configs/offline_validation.yaml

python3 thesis_offline_validation/scripts/02_run_dual_inference.py \
  --config thesis_offline_validation/configs/offline_validation.yaml

python3 thesis_offline_validation/scripts/03_build_targets.py \
  --config thesis_offline_validation/configs/offline_validation.yaml \
  --targets-config thesis_offline_validation/configs/targets.yaml

python3 thesis_offline_validation/scripts/04_extract_features_image.py \
  --config thesis_offline_validation/configs/offline_validation.yaml

python3 thesis_offline_validation/scripts/05_extract_features_detection.py \
  --config thesis_offline_validation/configs/offline_validation.yaml

python3 thesis_offline_validation/scripts/06_extract_features_temporal.py \
  --config thesis_offline_validation/configs/offline_validation.yaml

python3 thesis_offline_validation/scripts/07_extract_features_roi.py \
  --config thesis_offline_validation/configs/offline_validation.yaml

python3 thesis_offline_validation/scripts/08_build_master_table.py \
  --config thesis_offline_validation/configs/offline_validation.yaml \
  --feature-config thesis_offline_validation/configs/feature_sets.yaml \
  --targets-config thesis_offline_validation/configs/targets.yaml
```

## Assumptions

- This offline pipeline currently uses YOLOv8s as a configurable pseudo-reference for benefit labeling.
- `y_benefit_iou` is a pseudo-localization target derived from n/s agreement, not human ground truth.
- Grouped validation by `video_id` is assumed for later statistical work. This module prepares the schema but does not yet implement the statistics scripts.

