# 实时查看训练进度

## ✅ 训练已启动！

当前配置：
- **环境**: BipedalWalker-v3
- **训练代数**: 500代
- **种群大小**: 50
- **保存频率**: 每100代保存模型和视频
- **预计时间**: 40-60分钟

训练日志文件：
```
training_log_20251126_192504.txt
```

---

## 📊 实时查看训练进度

### 方法1：实时跟踪日志（推荐）

在终端运行：
```bash
cd "/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA"
tail -f training_log_20251126_192504.txt
```

按 `Ctrl+C` 退出查看

### 方法2：查看最新进度

```bash
tail -30 training_log_20251126_192504.txt
```

### 方法3：在IDE中打开日志文件

直接在VS Code或其他编辑器中打开：
```
/Users/cq123222/Desktop/25-26 Fall/生命科学导论/GA/training_log_20251126_192504.txt
```

---

## 📈 训练输出格式说明

每一代的输出示例：
```
[Gen 8/500] Eval: 10/50... Eval: 20/50... Eval: 30/50... Eval: 40/50... Eval: 50/50... NEW RECORD! Best: -3.35, Mean: -72.58, Std: 43.11 | Time: 3.4s
```

**含义**：
- `[Gen 8/500]`: 第8代/共500代
- `Eval: X/50`: 正在评估第X个个体（共50个）
- `NEW RECORD!`: 发现了更好的个体
- `Best: -3.35`: 本代最佳适应度
- `Mean: -72.58`: 平均适应度
- `Std: 43.11`: 标准差
- `Time: 3.4s`: 本代用时

---

## 🎯 适应度进展预期

| 代数 | 适应度范围 | 机器人行为 |
|------|-----------|-----------|
| 0-50 | -100 到 -50 | 学会站立，不立即倒下 |
| 50-100 | -50 到 0 | 开始尝试移动 |
| 100-200 | 0 到 50 | 产生行走动作 |
| 200-300 | 50 到 150 | 稳定向前移动 |
| 300-500 | 150+ | 流畅行走 |

**注意**：适应度达到100+就已经很好了！

---

## 💾 检查保存的文件

### 查看已保存的模型

```bash
ls -lh best_model*.npy
```

每100代会生成：
- `best_model_gen100.npy`
- `best_model_gen200.npy`
- 等等...

### 查看已生成的视频

```bash
ls -lh videos/*/
```

每100代会生成一个演示视频

### 查看训练曲线（中途查看）

```bash
conda activate ga-humanoid
python visualize.py --plot-only
```

---

## 🛑 停止训练

如果需要停止训练：

```bash
kill 60541
```

或者：
```bash
pkill -f train_with_video.py
```

**不用担心**：停止训练不会丢失已保存的检查点！

---

## ⏰ 预计时间线

- **每代**: 约3-5秒
- **100代**: 约6-10分钟
- **500代**: 约40-60分钟

当前进度可以通过日志中的 `[Gen X/500]` 查看。

---

## 🎥 训练完成后

训练结束后，会自动生成：

1. **最终模型**: `best_model.npy`
2. **检查点模型**: `best_model_gen100.npy`, `best_model_gen200.npy`, ...
3. **演示视频**: `videos/` 目录下
4. **训练统计**: `training_stats.npy`

### 查看训练结果

```bash
# 生成训练曲线图
conda activate ga-humanoid
python visualize.py --plot-only

# 运行最佳模型
python visualize.py --episodes 5

# 为所有检查点生成视频对比
python generate_videos.py
```

---

## 💡 常见问题

### Q: 怎么知道训练是否还在运行？

A: 查看进程：
```bash
ps aux | grep train_with_video.py | grep -v grep
```

如果看到进程，说明还在运行。

### Q: 适应度一直是负数正常吗？

A: 正常！BipedalWalker的适应度从-100左右开始，逐渐提升到正数。
   - 负数但绝对值在减小：正在进步！
   - 达到0以上：已经很好了！
   - 达到100+：非常好！

### Q: 看不到NEW RECORD怎么办？

A: 这很正常。有些代不会产生更好的个体，遗传算法会经历"停滞期"。继续训练通常能突破。

### Q: 可以同时做其他事情吗？

A: 可以！训练在后台运行，不影响其他操作。只是会占用一些CPU。

---

## 📞 快速命令参考

```bash
# 实时查看进度
tail -f training_log_20251126_192504.txt

# 查看最新30行
tail -30 training_log_20251126_192504.txt

# 查看进程
ps aux | grep train_with_video.py

# 停止训练
kill 60541

# 查看已保存的文件
ls -lh best_model*.npy
ls -lh videos/*/

# 中途查看训练曲线
conda activate ga-humanoid && python visualize.py --plot-only
```

---

**训练进行中，请耐心等待！可以随时用上述命令查看进度。** 🚀

