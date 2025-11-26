# 使用指南

## 快速开始

### 1. 安装依赖

**方式一：使用快速开始脚本（推荐）**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

**方式二：手动安装**
```bash
# 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证安装

运行简单的CartPole演示来验证算法是否正常工作：

```bash
python demo_cartpole.py
```

这个演示通常在1-2分钟内完成，如果运行成功，说明环境配置正确。

### 3. 训练3D人形机器人

**方式一：使用默认配置**
```bash
python train_humanoid.py
```

**方式二：使用配置文件（更灵活）**

首先编辑 `config.py` 调整参数，然后运行：
```bash
python train_simple.py
```

### 4. 查看训练结果

```bash
python visualize.py
```

## 详细说明

### 文件结构

```
GA/
├── neural_network.py       # 神经网络控制器
├── genetic_algorithm.py    # 遗传算法核心
├── train_humanoid.py       # 主训练程序（内置配置）
├── train_simple.py         # 简化训练程序（使用config.py）
├── visualize.py            # 可视化工具
├── config.py               # 配置文件
├── demo_cartpole.py        # CartPole演示
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明
├── USAGE.md               # 本文件
└── quick_start.sh         # 快速开始脚本
```

### 配置参数说明

编辑 `config.py` 可以调整以下参数：

#### 环境设置
- `ENV_NAME`: 环境名称
  - `'Humanoid-v4'`: 3D人形机器人（最复杂）
  - `'Walker2d-v4'`: 2D步行器
  - `'BipedalWalker-v3'`: 2D双足机器人（较简单，推荐初学）

#### 遗传算法参数
- `POPULATION_SIZE`: 种群大小（30-100）
  - 越大越容易找到好的解，但训练越慢
- `GENERATIONS`: 训练代数（50-200）
  - 需要足够的代数才能看到明显改进
- `MUTATION_RATE`: 变异率（0.1-0.3）
  - 太低：收敛快但可能陷入局部最优
  - 太高：探索多但收敛慢
- `CROSSOVER_RATE`: 交叉率（0.7-0.9）
  - 控制父代基因混合的频率
- `ELITE_RATIO`: 精英比例（0.05-0.15）
  - 最优个体直接保留到下一代的比例

#### 神经网络参数
- `HIDDEN_LAYERS`: 隐藏层结构
  - `[64, 32]`: 两层（推荐）
  - `[128, 64]`: 更复杂的网络
  - `[32]`: 单层（更快但可能性能较差）

#### 训练参数
- `MAX_STEPS`: 每集最大步数（500-1000）
- `RENDER_BEST`: 是否实时显示最佳个体（True/False）
- `RENDER_FREQUENCY`: 渲染频率（每N代渲染一次）

### 不同环境的难度对比

| 环境 | 维度 | 难度 | 建议代数 | 说明 |
|------|------|------|---------|------|
| CartPole-v1 | 4→2 | ⭐ | 20-50 | 平衡杆，入门级 |
| BipedalWalker-v3 | 24→4 | ⭐⭐⭐ | 100-300 | 2D双足行走 |
| Walker2d-v4 | 17→6 | ⭐⭐⭐⭐ | 200-500 | 2D行走器 |
| Humanoid-v4 | 376→17 | ⭐⭐⭐⭐⭐ | 500+ | 3D人形，最难 |

### 训练建议

#### 初次使用
1. 先运行 `demo_cartpole.py` 验证环境
2. 使用 `BipedalWalker-v3` 环境训练（更快看到效果）
3. 设置较小的参数：种群30，代数50

#### 快速测试
在 `config.py` 中设置：
```python
QUICK_TEST = True
```

#### 正式训练
- 使用 `Humanoid-v4` 环境
- 种群大小：50-100
- 训练代数：100-200+
- 预计时间：数小时到一天（取决于硬件）

### 查看训练进度

训练过程中会显示：
- 每代的最佳、平均、标准差适应度
- 发现新纪录时的提示
- 定期展示最佳个体的表现

### 保存和加载

#### 自动保存
- 每 `SAVE_FREQUENCY` 代自动保存检查点
- 文件名：`best_model_gen10.npy`, `best_model_gen20.npy` 等

#### 加载特定模型
```bash
python visualize.py --model best_model_gen50.npy
```

#### 仅查看训练曲线
```bash
python visualize.py --plot-only
```

#### 仅运行模型（不绘图）
```bash
python visualize.py --no-plot --episodes 10
```

### 故障排除

#### MuJoCo安装失败

**症状**: ImportError: No module named 'mujoco'

**解决方案**:
```bash
# macOS
brew install mujoco

# 或使用更简单的环境
# 在 config.py 中设置：
ENV_NAME = 'BipedalWalker-v3'
```

#### 训练太慢

**优化建议**:
1. 减小种群大小（如30）
2. 减少MAX_STEPS（如500）
3. 关闭实时渲染：`RENDER_BEST = False`
4. 使用更简单的环境
5. 减少隐藏层神经元数量

#### 无法看到可视化窗口

**可能原因**:
- 远程服务器没有显示
- macOS需要授予权限

**解决方案**:
- 本地运行
- 或设置 `RENDER_BEST = False`，训练后用 `visualize.py` 查看

#### 适应度不提升

**可能原因及解决**:
1. 陷入局部最优
   - 增加变异率：`MUTATION_RATE = 0.2`
   - 减小精英比例：`ELITE_RATIO = 0.05`
2. 种群多样性不足
   - 增加种群大小：`POPULATION_SIZE = 100`
3. 网络容量不足
   - 增加隐藏层：`HIDDEN_LAYERS = [128, 64]`
4. 训练时间不够
   - 继续训练更多代

### 期望结果

#### CartPole
- 10-20代：得分100-200
- 20-30代：得分400-500（接近完美）

#### BipedalWalker-v3
- 0-50代：机器人学会站立（得分-100到0）
- 50-100代：开始移动（得分0-100）
- 100-200代：稳定行走（得分100-250）
- 200+代：熟练行走（得分250+）

#### Humanoid-v4
- 0-20代：立即倒下（得分100-200）
- 20-50代：能站立片刻（得分200-500）
- 50-100代：保持平衡（得分500-2000）
- 100-200代：开始移动（得分2000-4000）
- 200+代：类似行走动作（得分4000+）

注意：Humanoid-v4 非常困难，可能需要数百代和大量调参才能获得良好效果。

### 进阶技巧

#### 使用自适应遗传算法

修改代码使用 `AdaptiveGeneticAlgorithm`：
```python
from genetic_algorithm import AdaptiveGeneticAlgorithm

ga = AdaptiveGeneticAlgorithm(...)
```

#### 调整初始化

修改 `neural_network.py` 中的初始化规模：
```python
params = create_random_params(param_count, scale=2.0)  # 增大初始规模
```

#### 自定义适应度函数

可以修改适应度计算，例如：
- 奖励前进距离
- 惩罚能量消耗
- 奖励保持直立

#### 并行评估

如果有多核CPU，可以使用 `multiprocessing` 并行评估种群。

### 示例训练命令

```bash
# 快速测试（5-10分钟）
python demo_cartpole.py

# 中等难度（30-60分钟）
# config.py: ENV_NAME='BipedalWalker-v3', GENERATIONS=100
python train_simple.py

# 完整训练（数小时）
# config.py: ENV_NAME='Humanoid-v4', GENERATIONS=200, POPULATION_SIZE=100
python train_simple.py
```

### 性能基准

在标准笔记本电脑上（Intel i5/i7）：
- CartPole: ~2秒/代（种群30）
- BipedalWalker: ~10秒/代（种群50）
- Humanoid: ~30-60秒/代（种群50）

GPU不会加速（因为使用NumPy而非深度学习框架）。

## 常见问题

**Q: 为什么不用强化学习算法如PPO？**

A: 本项目是遗传算法的教学演示。遗传算法是进化计算的代表，与基于梯度的强化学习是不同的范式。遗传算法的优势：
- 不需要梯度信息
- 可以处理非可微的问题
- 易于理解和实现
- 自然并行化

**Q: 遗传算法相比强化学习如何？**

A: 
- 简单环境：遗传算法表现不错
- 复杂环境：现代强化学习（如PPO、SAC）通常更高效
- 特殊场景：遗传算法在某些问题上仍有优势

**Q: 如何提高训练效率？**

A: 
1. 使用更简单的环境先验证
2. 合理设置种群大小和代数
3. 调整变异率和交叉率
4. 考虑使用自适应参数
5. 并行化评估过程

**Q: 可以用其他环境吗？**

A: 可以！只需修改 `ENV_NAME`，任何连续动作空间的Gymnasium环境都可以尝试。

## 相关资源

- [Gymnasium 文档](https://gymnasium.farama.org/)
- [遗传算法教程](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [NEAT算法](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)（更高级的进化算法）
- [OpenAI Evolution Strategies](https://openai.com/research/evolution-strategies)

## 许可与致谢

本项目仅供教育学习使用。

使用的库：
- Gymnasium（OpenAI Gym的继任者）
- NumPy
- Matplotlib
- MuJoCo（可选）

