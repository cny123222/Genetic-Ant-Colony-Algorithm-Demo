#!/bin/bash
# CMA-ES训练监控脚本

# 找到最新的CMA-ES日志
LOG_FILE=$(ls -t training_cmaes_*.txt 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到CMA-ES训练日志"
    exit 1
fi

echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'
echo "📊 CMA-ES训练进度监控"
echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'
echo "📝 日志文件: $LOG_FILE"
echo ""

# 检查进程
if [ -f cmaes_training.pid ]; then
    PID=$(cat cmaes_training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 训练进程运行中 (PID: $PID)"
    else
        echo "⚠️  训练进程已结束"
    fi
fi

echo ""
echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'
echo "📈 训练进度"
echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'

# 当前代数
CURRENT_GEN=$(grep -o '\[Gen [0-9]*/[0-9]*\]' "$LOG_FILE" | tail -1)
if [ -n "$CURRENT_GEN" ]; then
    echo "当前: $CURRENT_GEN"
else
    echo "当前: 初始化中..."
fi

# 最佳分数
echo ""
echo "🏆 最佳记录:"
grep "NEW RECORD" "$LOG_FILE" | tail -5

# 最新进度
echo ""
echo "📊 最近进度:"
grep -E '\[Gen [0-9]+/' "$LOG_FILE" | tail -10

# 时间估算
TOTAL_GENS=$(grep -o 'Gen [0-9]*/\([0-9]*\)' "$LOG_FILE" | head -1 | sed 's/.*\/\([0-9]*\).*/\1/')
CURRENT=$(grep -o 'Gen \([0-9]*\)/' "$LOG_FILE" | tail -1 | sed 's/Gen \([0-9]*\)\//\1/')

if [ -n "$TOTAL_GENS" ] && [ -n "$CURRENT" ] && [ "$CURRENT" -gt 0 ]; then
    # 计算已用时间
    START_TIME=$(grep "训练开始时间" "$LOG_FILE" | head -1 | awk '{print $2, $3}')
    if [ -n "$START_TIME" ]; then
        ELAPSED_SECONDS=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$START_TIME" +%s 2>/dev/null || echo 0) ))
        if [ $ELAPSED_SECONDS -gt 0 ]; then
            AVG_TIME_PER_GEN=$(( ELAPSED_SECONDS / CURRENT ))
            REMAINING_GENS=$(( TOTAL_GENS - CURRENT ))
            ETA_SECONDS=$(( AVG_TIME_PER_GEN * REMAINING_GENS ))
            
            echo ""
            echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'
            echo "⏱️  时间估算"
            echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'
            echo "已用时间: $(( ELAPSED_SECONDS / 60 )) 分钟"
            echo "平均每代: ${AVG_TIME_PER_GEN} 秒"
            echo "预计剩余: $(( ETA_SECONDS / 60 )) 分钟 ($(( ETA_SECONDS / 3600 )) 小时)"
        fi
    fi
fi

echo ""
echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'
echo "💡 提示:"
echo "  查看完整日志: tail -f $LOG_FILE"
echo "  查看视频: ls -la videos_cmaes/"
echo "  查看模型: ls -la models_cmaes/"
echo "=" | awk '{s=sprintf("%80s", " "); gsub(/ /, "=", s); print s}'

