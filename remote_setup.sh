#!/bin/bash
# Remote machine setup script for DMS-Raptor experiments
# Usage: ssh gchi@134.109.184.66 'bash -s' < remote_setup.sh

set -e
echo "=== DMS-Raptor Remote Setup ==="

# 1. Clone repo
cd ~
if [ -d "DMS_Clean" ]; then
    echo "Repo exists, pulling latest..."
    cd DMS_Clean && git pull
else
    echo "Cloning repo..."
    git clone https://github.com/YOUR_USERNAME/DMS_Clean.git
    cd DMS_Clean
fi

# 2. Create conda env (if not exists)
if ! conda env list | grep -q "dms-exp"; then
    echo "Creating conda environment..."
    conda create -n dms-exp python=3.10 -y
fi
conda activate dms-exp

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directory structure
mkdir -p models videos/glass_insulator_videos videos/porcelain_insulator_videos results

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Download videos:  bash download_videos.sh"
echo "  2. Copy models to:   ~/DMS_Clean/models/"
echo "  3. Run experiments:  python dms_experiment.py run-all --device cuda"
