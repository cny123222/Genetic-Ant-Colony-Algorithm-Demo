#!/bin/bash
# CMA-ES训练启动脚本

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_cmaes_${TIMESTAMP}.txt"

echo "🚀 启动CMA-ES训练..."
echo "📝 日志文件: ${LOG_FILE}"
echo ""

# 激活conda环境并运行
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ga-humanoid

# 使用unbuffered输出
nohup python -u train_cmaes.py > "${LOG_FILE}" 2>&1 &

PID=$!
echo "✅ 训练已在后台启动"
echo "📊 进程ID: ${PID}"
echo "📝 日志文件: ${LOG_FILE}"
echo ""
echo "查看实时日志: tail -f ${LOG_FILE}"
echo "停止训练: kill ${PID}"
echo ""
echo "PID ${PID} saved to cmaes_training.pid"
echo "${PID}" > cmaes_training.pid

