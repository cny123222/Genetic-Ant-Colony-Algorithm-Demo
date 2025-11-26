#!/bin/bash

# 实时显示训练进度的脚本
# 输出会同时显示在终端和保存到日志文件

echo "=================================================="
echo "Starting GA Training with Real-time Output"
echo "=================================================="
echo ""

cd "/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA"

# 激活conda环境并运行（输出到终端和文件）
conda run -n ga-humanoid python -u train_with_video.py 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

