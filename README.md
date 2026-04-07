# DMS-Raptor: Dynamic Model Switching for Real-Time UAV Inspection

**Author:** Ganapathi Teja Chirravuri
**University:** TU Chemnitz, Faculty of Computer Science

Single-file experiment pipeline for the DMS thesis. Runs all 8 switching policies across 12 UAV inspection videos, generates 50+ plots, 30 validation images per policy, and a complete analysis report.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place files
#    models/  -> glass_y8n_fast.pt, glass_y8s_accurate.pt, etc.
#    videos/glass_insulator_videos/  -> *.mp4
#    videos/porcelain_insulator_videos/  -> *.mp4

# 3. Run everything (CPU)
python dms_experiment.py run-all --device cpu --max-frames 500

# 4. Run everything (GPU)
python dms_experiment.py run-all --device cuda
```

## Commands

| Command | What it does |
|---------|-------------|
| `run-all` | Detection + Timing + Validation + Overnight + Plots + Report |
| `detection` | Detection quality (both models on every frame, s_only as oracle) |
| `timing` | Repeated timing trials (N trials per policy) |
| `validation` | Frame-level validation (30 n-routed + 30 s-routed images per policy) |
| `overnight` | Full pipeline (all 8 policies x all videos, with traces & videos) |
| `plots` | Generate all figures from existing results |
| `report` | Generate thesis analysis report from existing results |

## The 8 Policies

| Policy | Type | Signal | T_scene |
|--------|------|--------|---------|
| n_only | Baseline | Always YOLOv8n | 0ms |
| s_only | Baseline | Always YOLOv8s | 0ms |
| entropy_only | Proactive | Shannon entropy vs median | ~3ms |
| combined | Proactive | C-score (alpha*L + (1-a)*H) single threshold | ~3ms |
| combined_hyst | Proactive | C-score with hysteresis band | ~3ms |
| conf_ema | Reactive | Dual-EMA on detection confidence | 0ms |
| niqe_switch | Proactive | NR-IQA quality score | ~3ms |
| multi_proxy | Proactive | 4-proxy weighted composite | ~8ms |

## Output Structure

```
results/
  grand_summary.json
  THESIS_ANALYSIS_REPORT.md
  figures/                          # 50+ publication-quality PNGs
  detection_quality/<video>/        # Precision/Recall/F1 per policy
  timing_trials/<video>/            # Multi-trial timing statistics
  frame_validation/<video>/
    <policy>/n_routed/              # 30 side-by-side images
    <policy>/s_routed/              # 30 side-by-side images
    <policy>/grid_*.png             # 5x6 thumbnail grids
  overnight/<material>/<video>/
    <policy>_summary.json           # Full run metrics
    <policy>_trace.csv              # Per-frame data
    <policy>_annotated.mp4          # Annotated video
```

---

## Complete Remote Workflow

### Step 1: Push to GitHub (local machine)

```bash
cd G:/Teja_Master_Thesis/DMS_Clean
git init
git add .
git commit -m "DMS-Raptor clean experiment pipeline"
git remote add origin https://github.com/YOUR_USERNAME/DMS_Clean.git
git push -u origin main
```

### Step 2: Setup remote machine

```bash
ssh gchi@134.109.184.66

# Clone
cd ~
git clone https://github.com/YOUR_USERNAME/DMS_Clean.git
cd DMS_Clean

# Environment
conda create -n dms-exp python=3.10 -y
conda activate dms-exp
pip install -r requirements.txt

mkdir -p models videos/glass_insulator_videos videos/porcelain_insulator_videos
```

### Step 3: Transfer videos & models to remote

**Option A: SCP from local** (run from local Git Bash / PowerShell)
```bash
scp G:/Teja_Master_Thesis/MT_Chirravuri/models/*.pt gchi@134.109.184.66:~/DMS_Clean/models/
scp G:/Teja_Master_Thesis/MT_Chirravuri/input_videos/glass_insulator_videos/*.mp4 gchi@134.109.184.66:~/DMS_Clean/videos/glass_insulator_videos/
scp G:/Teja_Master_Thesis/MT_Chirravuri/input_videos/porcelain_insulator_videos/*.mp4 gchi@134.109.184.66:~/DMS_Clean/videos/porcelain_insulator_videos/
```

**Option B: Google Drive -> Remote** (run on remote)
```bash
# Install rclone and configure Google Drive
curl https://rclone.org/install.sh | sudo bash
rclone config   # Add remote named "gdrive", type: Google Drive

# Download
rclone copy "gdrive:MT_Chirravuri/models" models/ --progress
rclone copy "gdrive:MT_Chirravuri/input_videos/glass_insulator_videos" videos/glass_insulator_videos/ --progress
rclone copy "gdrive:MT_Chirravuri/input_videos/porcelain_insulator_videos" videos/porcelain_insulator_videos/ --progress
```

### Step 4: Run experiments on remote (GPU)

```bash
ssh gchi@134.109.184.66
cd ~/DMS_Clean
conda activate dms-exp

# Quick test first (500 frames, ~10 min)
python dms_experiment.py run-all --device cuda --max-frames 500

# Full run (all frames, all videos -- overnight)
nohup python dms_experiment.py run-all --device cuda \
    --annotate-policies conf_ema combined_hyst niqe_switch \
    > run.log 2>&1 &

# Monitor progress
tail -f run.log
```

### Step 5: Upload results to Google Drive (from remote)

```bash
# Option A: rclone
rclone copy results/ "gdrive:DMS_Experiment_Results/" --progress

# Option B: tar + scp
tar -czf dms_results.tar.gz results/
# Then from LOCAL:
scp gchi@134.109.184.66:~/DMS_Clean/dms_results.tar.gz G:/Teja_Master_Thesis/DMS_Clean/
```

### Step 6: Download to local (from Google Drive or SCP)

```bash
# Option A: rclone on local
rclone copy "gdrive:DMS_Experiment_Results" G:/Teja_Master_Thesis/DMS_Clean/results/ --progress

# Option B: SCP directly
scp -r gchi@134.109.184.66:~/DMS_Clean/results/ G:/Teja_Master_Thesis/DMS_Clean/results/
```

### Step 7: Regenerate plots locally (optional)

```bash
cd G:/Teja_Master_Thesis/DMS_Clean
python dms_experiment.py plots --results-dir results
python dms_experiment.py report --results-dir results
```
