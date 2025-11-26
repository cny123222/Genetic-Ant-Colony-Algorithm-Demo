#!/bin/bash

# 启动训练脚本
# 训练500代，每100代保存一次模型和视频

echo "=================================================="
echo "Starting GA Training for BipedalWalker-v3"
echo "=================================================="
echo "Configuration:"
echo "  - Generations: 500"
echo "  - Population: 50"
echo "  - Save Frequency: 100 generations"
echo "  - Video Frequency: 100 generations"
echo "=================================================="
echo ""

cd "/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA"

# 运行训练（输出到控制台和文件）
conda run -n ga-humanoid python train_with_video.py 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Results saved in:"
echo "  - best_model.npy (final model)"
echo "  - best_model_gen*.npy (checkpoints)"
echo "  - videos/ (demo videos)"
echo "  - training_stats.npy (statistics)"
echo "  - training_progress.png (plot)"
echo "=================================================="

