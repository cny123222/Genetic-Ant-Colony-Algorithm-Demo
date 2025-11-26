# 遗传算法训练3D人形机器人行走 🤖

使用遗传算法（Genetic Algorithm）进化神经网络参数，训练3D人形机器人学会行走。

这是一个完整的教学演示项目，展示了如何使用进化计算方法解决强化学习问题。

## 项目演示

![训练效果演示](https://img.shields.io/badge/demo-genetic_algorithm-blue)

本项目支持多个环境：
- **Humanoid-v4**: 3D人形机器人（17个自由度）
- **BipedalWalker-v3**: 2D双足机器人（推荐入门）
- **Walker2d-v4**: 2D步行器
- **CartPole-v1**: 平衡杆（快速验证）

## 功能特点

- 🧬 **遗传算法**: 完整实现选择、交叉、变异操作
- 🤖 **3D机器人**: 支持复杂的人形机器人环境
- 📊 **实时可视化**: 每代展示最佳个体表现
- 💾 **自动保存**: 定期保存模型和训练统计
- ⚙️ **灵活配置**: 通过配置文件轻松调整参数
- 📈 **训练分析**: 自动生成训练曲线图表
- 🚀 **快速开始**: 一键安装脚本和演示程序

## 快速开始

### 方法一：使用一键安装脚本（推荐）

```bash
chmod +x quick_start.sh
./quick_start.sh
```

### 方法二：手动安装

```bash
# 1. 克隆或下载项目
cd GA

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 测试环境
python test_modules.py

# 5. 运行演示
python demo_cartpole.py
```

### MuJoCo 安装说明（可选）

MuJoCo用于Humanoid-v4环境。如果想使用3D人形机器人，需要安装：

**macOS:**
```bash
brew install mujoco
```

**Linux/Windows:**
```bash
pip install mujoco  # 通常自动安装
```

**如果安装困难**：使用BipedalWalker-v3环境（无需MuJoCo）。

## 使用方法

### 1. 验证安装

```bash
python test_modules.py
```

这会检查所有依赖和环境是否正确安装。

### 2. 快速演示（1-2分钟）

```bash
python demo_cartpole.py
```

在CartPole环境上快速验证算法是否正常工作。

### 3. 训练模型

**简单训练（使用配置文件）：**
```bash
# 编辑 config.py 调整参数
python train_simple.py
```

**标准训练：**
```bash
python train_humanoid.py
```

**配置参数示例（config.py）：**
```python
ENV_NAME = 'BipedalWalker-v3'  # 或 'Humanoid-v4'
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.15
HIDDEN_LAYERS = [64, 32]
```

### 4. 可视化结果

```bash
# 查看训练曲线和运行模型
python visualize.py

# 仅查看训练曲线
python visualize.py --plot-only

# 运行多个episodes
python visualize.py --episodes 10
```

## 文件结构

```
GA/
├── 核心模块
│   ├── neural_network.py       # 神经网络控制器
│   ├── genetic_algorithm.py    # 遗传算法核心逻辑
│   └── config.py              # 配置文件
│
├── 训练脚本
│   ├── train_humanoid.py      # 主训练程序（内置配置）
│   ├── train_simple.py        # 简化训练程序（使用config.py）
│   └── demo_cartpole.py       # CartPole演示（快速验证）
│
├── 工具脚本
│   ├── visualize.py           # 可视化工具（查看结果）
│   ├── test_modules.py        # 测试脚本（环境检查）
│   └── quick_start.sh         # 快速安装脚本
│
├── 文档
│   ├── README.md             # 项目说明（本文件）
│   └── USAGE.md              # 详细使用指南
│
├── 配置
│   ├── requirements.txt       # Python依赖列表
│   └── .gitignore            # Git忽略文件
│
└── 生成文件（训练后）
    ├── best_model.npy         # 最佳模型
    ├── best_model_genN.npy    # 检查点模型
    ├── training_stats.npy     # 训练统计
    └── training_progress.png  # 训练曲线图
```

## 工作原理

### 遗传算法流程

```
初始化种群（随机参数）
    ↓
┌─→ 评估适应度（在环境中运行）
│       ↓
│   选择优秀个体（锦标赛选择）
│       ↓
│   交叉产生子代（参数混合）
│       ↓
│   变异（添加噪声）
│       ↓
│   精英保留（保留最优个体）
│       ↓
└─ 新一代种群（重复或终止）
```

**关键步骤：**
1. **初始化**: 随机生成N个神经网络参数向量
2. **评估**: 每个网络控制机器人，累计奖励作为适应度
3. **选择**: 锦标赛选择 - 随机选3个，保留最好的
4. **交叉**: 两个父代参数按概率混合
5. **变异**: 以一定概率对参数添加高斯噪声
6. **精英保留**: 前10%最优个体直接进入下一代

### 神经网络结构

不同环境的网络规模：

| 环境 | 输入维度 | 隐藏层 | 输出维度 | 总参数 |
|------|---------|--------|---------|--------|
| CartPole-v1 | 4 | [16] | 1 | ~81 |
| BipedalWalker-v3 | 24 | [64, 32] | 4 | ~3,748 |
| Humanoid-v4 | 376 | [64, 32] | 17 | ~26,705 |

**默认结构（Humanoid-v4）：**
- 输入层: 376维（关节角度、速度等）
- 隐藏层1: 64个神经元（tanh激活）
- 隐藏层2: 32个神经元（tanh激活）
- 输出层: 17维（关节扭矩控制）

所有参数被扁平化为一维向量供遗传算法操作。

## 预期效果

- 初期（0-20代）: 机器人基本无法站立，会立即倒下
- 中期（20-50代）: 机器人学会保持平衡，可以站立一段时间
- 后期（50代以上）: 机器人开始学会向前移动，产生类似行走的动作

完整训练可能需要数百代才能达到稳定行走。

## 参数调优建议

- **种群过小**: 增加 `POPULATION_SIZE` 到 100+
- **收敛太快**: 增加 `MUTATION_RATE` 或减小 `ELITE_RATIO`
- **进展太慢**: 减小 `MUTATION_RATE` 或增加 `ELITE_RATIO`
- **想看更快效果**: 使用更简单的环境如 `BipedalWalker-v3`

## 故障排除

### MuJoCo 安装失败
尝试使用更简单的环境：在 `train_humanoid.py` 中将环境改为：
```python
env = gym.make('BipedalWalker-v3', render_mode='human')
```

### 训练太慢
- 减小种群大小
- 减少每集的最大步数
- 使用更简单的环境

### 程序崩溃
- 确保安装了所有依赖
- 检查 Python 版本（建议 3.8+）
- 尝试在虚拟环境中运行

## 许可

本项目仅供教育和学习使用。

