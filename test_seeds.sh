#!/bin/bash
# 测试不同随机种子，找到好的

SEEDS=(42 123 456 789 1024 2048 4096 8192)

for seed in "${SEEDS[@]}"; do
    echo "Testing seed: $seed"
    
    # 修改config中的种子
    sed -i.bak "s/RANDOM_SEED = [0-9]*/RANDOM_SEED = $seed/" config.py
    
    # 快速训练（只跑200代测试）
    # TODO: 运行并记录结果
done
