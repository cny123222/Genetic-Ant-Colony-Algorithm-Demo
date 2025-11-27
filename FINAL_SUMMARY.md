# 训练总结 - 所有轮次对比

## Round 4 → Round 7 的改进过程

### Round 4（基准）
- **参数**: 180个体, 400代, 变异0.25, 随机地形
- **结果**: 最佳153.29 (Gen 336), 后期停滞64代
- **问题**: 
  1. 视频保存bug（用错误的seed）
  2. 代数可能不够

### Round 5（失败尝试）
- **改动**: 增大种群200，增加代数500，更高变异0.30
- **问题**: 过度调整，效果未验证就停止

### Round 6（固定地形实验 - 失败）
- **改动**: 使用5个固定地形 `[42, 123, 456, 789, 1024]`
- **结果**: 最佳-36.10（负数！）
- **问题**: 5个固定地形都太难，平均分永远是负数
- **教训**: 固定地形需要仔细选择简单的地形

### Round 7（当前 - 正确方向）✓
- **配置**: Round 4参数 + 600代 + 修复bug
- **改动**:
  1. ✅ 修复视频seed bug（现在用最佳个体的真实seed）
  2. ✅ 增加训练代数：400 → 600 (+50%)
  3. ✅ 保持Round 4已验证的参数
  4. ✅ 使用随机地形（不用固定地形）

## 关键Bug修复

### Bug: 视频保存用错误的seed

**问题**：
```python
# 之前的错误代码
for i in range(180):
    fitness, seed = evaluate(individual[i])
    # seed被覆盖180次

save_video(seed)  # 这是第180个个体的seed，不是最佳个体的！
```

**影响**：
- Round 4的视频可能不准确
- NEW RECORD视频显示的不是真实表现

**修复**：
```python
# 修复后
seeds = []
for i in range(180):
    fitness, seed = evaluate(individual[i])
    seeds.append(seed)

best_idx = argmax(fitness)
save_video(seeds[best_idx])  # ✓ 使用最佳个体的seed
```

## 当前配置 (Round 7)

```python
# 遗传算法
POPULATION_SIZE = 180
GENERATIONS = 600          # ← 比Round 4多200代
MUTATION_RATE = 0.25
MUTATION_SCALE = 0.5
CROSSOVER_RATE = 0.75
ELITE_RATIO = 0.015        # 3个精英
TOURNAMENT_SIZE = 5

# 网络
HIDDEN_LAYERS = [256, 128, 64]  # 47K参数

# 训练
USE_FIXED_TERRAIN = False  # 随机地形
VIDEO_ON_NEW_RECORD = True # 每次NEW RECORD录视频
```

## 预期结果

### 成功标准
- ✅ 最佳Fitness > 200
- ✅ 视频中机器人能走到终点
- ✅ 突破Round 4的153.29

### 时间估算
```
600代 × 180个体 × ~12秒/代 = ~2小时
```

### 如果达到200+
说明：
1. Round 4参数是好的
2. 只是需要更多训练时间
3. 视频bug修复后能真实反映进步

### 如果仍然<200
可能原因：
1. 需要调整参数（更大变异/网络）
2. 环境太难，需要换简单环境
3. GA方法本身的限制

## 监控要点

1. **突破频率**: 应该持续有NEW RECORD
2. **停滞检测**: 如果超过100代无突破需警惕
3. **视频质量**: 每次NEW RECORD视频应该比前一次更好
4. **最终分数**: 目标200+

## 下一步计划

### 如果Round 7成功 (>200)
1. Git commit保存最佳模型
2. 写实验报告
3. 展示训练视频对比

### 如果Round 7不成功 (<200)
方案A: 继续增加代数到1000
方案B: 尝试更大的网络 [512, 256, 128]
方案C: 换成更简单的环境测试方法
方案D: 考虑其他RL方法（PPO等）

## 学到的经验

1. **参数调整需谨慎**: Round 5一次改太多，无法判断哪个有效
2. **固定地形需精选**: Round 6证明随意选地形可能都太难
3. **Bug影响大**: 视频bug导致无法准确评估进展
4. **基准很重要**: Round 4虽然不完美，但是可靠的基准
5. **增量改进**: Round 7只改两个变量（代数+bug修复）

## 文件说明

- `train_with_video.py`: 主训练脚本（已修复bug）
- `config.py`: 配置文件（Round 7参数）
- `training_round7_*.txt`: 当前训练日志
- `videos/gen_XXX/`: NEW RECORD视频（按代数）
- `best_model_gen*.npy`: 每100代保存的模型
- `monitor.sh`: 监控脚本

## 使用说明

监控训练：
```bash
./monitor.sh
```

实时查看日志：
```bash
tail -f training_round7_*.txt
```

查看最新视频：
```bash
ls -lt videos/ | head -5
open videos/gen_XXX/
```

停止训练：
```bash
pkill -f train_with_video.py
```

