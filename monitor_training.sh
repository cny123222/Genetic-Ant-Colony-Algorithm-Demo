#!/bin/bash

# 监控训练进度脚本

echo "=================================================="
echo "Training Progress Monitor"
echo "=================================================="
echo ""

LOG_FILE="/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA/training_output.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Training log not found: $LOG_FILE"
    echo "Training may not have started yet."
    exit 1
fi

echo "Latest training progress (last 30 lines):"
echo "--------------------------------------------------"
tail -30 "$LOG_FILE"
echo "--------------------------------------------------"
echo ""

# 统计信息
TOTAL_GENS=$(grep -c "^\[Gen" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Completed Generations: $TOTAL_GENS / 500"

# 检查是否有新纪录
RECORDS=$(grep -c "NEW RECORD" "$LOG_FILE" 2>/dev/null || echo "0")
echo "New Records Found: $RECORDS"

# 检查保存的模型
MODELS=$(ls -1 best_model_gen*.npy 2>/dev/null | wc -l)
echo "Saved Checkpoints: $MODELS"

# 检查视频
VIDEOS=$(find videos/ -name "*.mp4" 2>/dev/null | wc -l)
echo "Generated Videos: $VIDEOS"

echo ""
echo "To view full log: tail -f $LOG_FILE"
echo "To stop training: pkill -f train_with_video.py"
echo "=================================================="

