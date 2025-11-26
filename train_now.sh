#!/bin/bash

# 正确的训练启动脚本 - 先激活环境再运行

echo "=================================================="
echo "Starting GA Training with Real-time Output"
echo "=================================================="
echo "Configuration:"
echo "  - Environment: BipedalWalker-v3"
echo "  - Generations: 500"
echo "  - Population: 50"
echo "  - Save every: 100 generations"
echo "=================================================="
echo ""

cd "/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA"

# 先激活conda环境，然后运行python
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ga-humanoid
python -u train_with_video.py 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

