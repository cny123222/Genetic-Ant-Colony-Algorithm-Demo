#!/bin/bash
# Round 5 训练监控脚本

LOG_FILE=$(ls -t training_round5_*.txt 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "============================================================"
echo "🎯 Round 5 训练监控"
echo "============================================================"
echo ""

# 当前代数
CURRENT_GEN=$(tail -1 "$LOG_FILE" | grep -oE "Gen [0-9]+/500" | head -1)
echo "📊 当前进度: $CURRENT_GEN"
echo ""

# 统计NEW RECORD
RECORD_COUNT=$(grep -c "NEW RECORD" "$LOG_FILE")
echo "🏆 突破次数: $RECORD_COUNT 次"
echo ""

# 最近5次突破
echo "📈 最近突破记录:"
grep "NEW RECORD" "$LOG_FILE" | tail -5 | while read line; do
    GEN=$(echo "$line" | grep -oE "Gen [0-9]+")
    FITNESS=$(echo "$line" | grep -oE "Best: -?[0-9]+\.[0-9]+" | cut -d' ' -f2)
    echo "  - $GEN: fitness=$FITNESS"
done
echo ""

# 最佳记录
BEST_RECORD=$(grep "NEW RECORD" "$LOG_FILE" | tail -1)
BEST_GEN=$(echo "$BEST_RECORD" | grep -oE "Gen [0-9]+")
BEST_FITNESS=$(echo "$BEST_RECORD" | grep -oE "Best: -?[0-9]+\.[0-9]+" | cut -d' ' -f2)
echo "🥇 当前最佳: $BEST_GEN, fitness=$BEST_FITNESS"
echo ""

# 最近10代的表现
echo "📉 最近10代表现:"
tail -20 "$LOG_FILE" | grep -E "Gen [0-9]+/500.*Best:" | tail -10 | while read line; do
    GEN=$(echo "$line" | grep -oE "Gen [0-9]+")
    FITNESS=$(echo "$line" | grep -oE "Best: -?[0-9]+\.[0-9]+" | cut -d' ' -f2)
    MEAN=$(echo "$line" | grep -oE "Mean: -?[0-9]+\.[0-9]+" | cut -d' ' -f2)
    if echo "$line" | grep -q "NEW RECORD"; then
        echo "  - $GEN: fitness=$FITNESS, mean=$MEAN ⭐ NEW RECORD"
    else
        echo "  - $GEN: fitness=$FITNESS, mean=$MEAN"
    fi
done
echo ""

# 视频数量
VIDEO_COUNT=$(ls -d videos/gen_* 2>/dev/null | wc -l | tr -d ' ')
echo "🎬 已生成视频: $VIDEO_COUNT 个"
echo ""

# 训练时长
START_TIME=$(head -30 "$LOG_FILE" | grep "开始时间" | cut -d' ' -f2-)
if [ -n "$START_TIME" ]; then
    echo "⏰ 开始时间: $START_TIME"
fi

# 预估完成时间（简单估算）
if [ -n "$CURRENT_GEN" ]; then
    CURRENT_NUM=$(echo "$CURRENT_GEN" | grep -oE "^Gen [0-9]+" | grep -oE "[0-9]+")
    if [ "$CURRENT_NUM" -gt 10 ]; then
        TOTAL_LINES=$(wc -l < "$LOG_FILE")
        AVG_LINES_PER_GEN=$((TOTAL_LINES / CURRENT_NUM))
        REMAINING_GENS=$((500 - CURRENT_NUM))
        # 粗略估算（假设每代约20秒）
        REMAINING_MINS=$((REMAINING_GENS * 20 / 60))
        echo "⏳ 预计剩余时间: ~${REMAINING_MINS}分钟 (粗略估算)"
    fi
fi

echo ""
echo "============================================================"
echo "💡 提示:"
echo "  - 实时查看日志: tail -f $LOG_FILE"
echo "  - 查看视频: open videos/gen_029/"
echo "  - 停止训练: pkill -f train_with_video.py"
echo "============================================================"

