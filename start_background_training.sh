#!/bin/bash

# 后台启动训练脚本

LOG_FILE="/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA/training_log_$(date +%Y%m%d_%H%M%S).txt"

echo "=================================================="
echo "Starting Background Training"
echo "=================================================="
echo "Log file: $LOG_FILE"
echo ""

cd "/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA"

# 启动后台训练
nohup bash -c "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate ga-humanoid && python -u train_with_video.py" > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "=================================================="

