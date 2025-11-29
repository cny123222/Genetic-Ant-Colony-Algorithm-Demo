# 遗传算法演示项目

## 项目结构

```
demos/
├── cartpole/           # CartPole平衡问题演示
├── course_selection/   # 选课问题（0-1背包）演示
└── walker_attempts/    # BipedalWalker训练记录（参考）
```

## 三个演示

### 1. CartPole 平衡问题
- **位置**: `demos/cartpole/`
- **问题**: 控制杆子保持平衡
- **结果**: 500分满分
- **运行**: `cd demos/cartpole && python train_with_video.py`
- **配置**: 修改 `config.py`

### 2. 选课问题（0-1背包）
- **位置**: `demos/course_selection/`
- **问题**: 200门课，15%时间预算，最大化收获
- **结果**: 1119 → 1361分（+21.6%提升）
- **运行**: `python demos/course_selection/course_selection_ga.py`
- **特点**: 1000代进化，清晰的优化曲线

### 3. Walker训练记录
- **位置**: `demos/walker_attempts/`
- **说明**: BipedalWalker训练尝试记录（参考用）

## 核心模块

- `genetic_algorithm.py` - 遗传算法核心实现
- `neural_network.py` - 神经网络控制器
- `train_with_video.py` - 训练脚本（支持视频录制）
- `config.py` - 配置文件
- `visualize.py` - 可视化工具

## 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate ga-humanoid

# 或使用pip
pip install -r requirements.txt
```

## 依赖

- Python 3.10
- gymnasium[box2d]
- numpy
- matplotlib
- mujoco (可选，用于3D环境)

