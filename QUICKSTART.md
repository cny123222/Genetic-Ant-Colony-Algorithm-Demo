# 快速开始指南 🚀

## 3分钟快速开始

### 步骤 1: 安装依赖（1分钟）

```bash
# 进入项目目录
cd GA

# 运行安装脚本
chmod +x quick_start.sh
./quick_start.sh
```

或手动安装：
```bash
pip install gymnasium numpy matplotlib
```

### 步骤 2: 验证安装（30秒）

```bash
python test_modules.py
```

看到"所有测试通过"就可以继续了！

### 步骤 3: 运行演示（1-2分钟）

```bash
python demo_cartpole.py
```

你会看到一个平衡杆游戏，机器人会在几十代内学会保持平衡。

## 接下来做什么？

### 选项 A: 训练2D机器人（30分钟，推荐）

编辑 `config.py`：
```python
ENV_NAME = 'BipedalWalker-v3'  # 改为2D环境
GENERATIONS = 100
POPULATION_SIZE = 50
```

运行：
```bash
python train_simple.py
```

### 选项 B: 训练3D人形机器人（数小时）

保持默认配置，直接运行：
```bash
python train_simple.py
```

注意：需要安装MuJoCo，可能需要较长时间训练。

### 选项 C: 使用快速测试模式（5-10分钟）

编辑 `config.py`：
```python
QUICK_TEST = True  # 启用快速测试
```

运行：
```bash
python train_simple.py
```

## 查看训练结果

训练完成后：
```bash
python visualize.py
```

这会：
1. 显示训练曲线图
2. 运行训练好的模型
3. 展示机器人的行走效果

## 常见问题

### Q: 依赖安装失败？

**A**: 分步安装：
```bash
pip install numpy
pip install matplotlib  
pip install gymnasium
# MuJoCo是可选的
pip install mujoco
```

### Q: MuJoCo安装失败？

**A**: 使用更简单的环境（不需要MuJoCo）：

在 `config.py` 中设置：
```python
ENV_NAME = 'BipedalWalker-v3'
```

### Q: 训练太慢？

**A**: 
1. 减小种群：`POPULATION_SIZE = 30`
2. 减少代数：`GENERATIONS = 50`
3. 关闭渲染：`RENDER_BEST = False`
4. 使用更简单的环境

### Q: 看不到可视化窗口？

**A**: 
- 确保不是在远程服务器上
- macOS可能需要授权
- 可以设置 `RENDER_BEST = False`，训练后再查看

## 文件说明

### 需要运行的：
- `demo_cartpole.py` - 快速演示
- `train_simple.py` - 简单训练（推荐）
- `train_humanoid.py` - 完整训练
- `visualize.py` - 查看结果

### 需要配置的：
- `config.py` - 调整参数

### 需要阅读的：
- `README.md` - 项目说明
- `USAGE.md` - 详细指南

### 不需要改动的：
- `neural_network.py` - 核心模块
- `genetic_algorithm.py` - 核心模块

## 推荐学习路径

### 第1天：了解基础
1. 运行 `test_modules.py` 检查环境
2. 运行 `demo_cartpole.py` 看演示
3. 阅读 `README.md` 了解原理

### 第2天：简单训练
1. 修改 `config.py` 使用 BipedalWalker
2. 运行 `train_simple.py` 训练
3. 用 `visualize.py` 查看结果

### 第3天：深入实验
1. 尝试不同参数组合
2. 比较不同环境
3. 阅读 `USAGE.md` 学习进阶技巧

### 第4天：挑战3D
1. 配置 Humanoid-v4 环境
2. 长时间训练
3. 分析训练曲线

## 参数调优速查

### 训练太慢？
```python
POPULATION_SIZE = 30  # 减小
GENERATIONS = 50      # 减少
MAX_STEPS = 500       # 减少
RENDER_BEST = False   # 关闭
```

### 效果不好？
```python
POPULATION_SIZE = 100      # 增大
GENERATIONS = 200          # 增加
MUTATION_RATE = 0.2        # 增加探索
HIDDEN_LAYERS = [128, 64]  # 更大网络
```

### 想快速测试？
```python
QUICK_TEST = True  # 在config.py中启用
```

## 预期效果时间表

### CartPole（demo_cartpole.py）
- 0-10代：学会基本平衡（100-200分）
- 10-30代：接近完美（400-500分）

### BipedalWalker-v3
- 0-20代：站立（-100 到 0分）
- 20-50代：尝试移动（0-50分）
- 50-100代：开始行走（50-150分）
- 100+代：稳定行走（150+分）

### Humanoid-v4
- 0-20代：立即倒下（100-300分）
- 20-50代：短暂站立（300-1000分）
- 50-100代：保持平衡（1000-3000分）
- 100+代：开始移动（3000+分）

## 获取帮助

1. 查看 `USAGE.md` 详细文档
2. 检查 `README.md` 故障排除部分
3. 运行 `test_modules.py` 诊断问题

## 成功标志 ✅

你知道项目运行成功如果：
- ✅ `test_modules.py` 显示"所有测试通过"
- ✅ `demo_cartpole.py` 能看到平衡杆动画
- ✅ 训练时能看到适应度逐渐提升
- ✅ `visualize.py` 能展示训练好的模型

祝训练愉快！🎉

