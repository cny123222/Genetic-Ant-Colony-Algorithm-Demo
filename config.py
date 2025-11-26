"""
配置文件
集中管理所有训练参数
"""

# ==================== 环境配置 ====================
ENV_NAME = 'BipedalWalker-v3'  # 环境名称: 'Humanoid-v4', 'BipedalWalker-v3', 'Walker2d-v4'
FALLBACK_ENV = 'BipedalWalker-v3'  # 备选环境

# ==================== 遗传算法参数 ====================
POPULATION_SIZE = 150       # 种群大小 - 进一步增加
GENERATIONS = 300           # 训练代数 - 减少代数，增加质量
MUTATION_RATE = 0.20        # 变异率 - 保持较高但不过分
MUTATION_SCALE = 0.4        # 变异幅度 - 中等
CROSSOVER_RATE = 0.8        # 交叉率 - 恢复
ELITE_RATIO = 0.02          # 精英比例 - 极少精英（150*0.02=3个）
TOURNAMENT_SIZE = 7         # 锦标赛大小 - 更强选择压力

# ==================== 神经网络参数 ====================
HIDDEN_LAYERS = [128, 64, 32]  # 更大更深的网络 - 增强表达能力

# ==================== 训练参数 ====================
MAX_STEPS = 1000            # 每个episode的最大步数
RENDER_BEST = False         # 是否渲染最佳个体（改为False加快训练）
RENDER_FREQUENCY = 1        # 每隔几代渲染一次（设为1表示每代都渲染）
SAVE_FREQUENCY = 50         # 每隔几代保存一次模型（改为50代）
VIDEO_FREQUENCY = 50        # 每隔几代保存一次视频（改为50代）

# ==================== 快速测试配置 ====================
# 如果想快速测试，可以使用这些参数
QUICK_TEST = False

if QUICK_TEST:
    POPULATION_SIZE = 20
    GENERATIONS = 20
    MAX_STEPS = 500
    SAVE_FREQUENCY = 5
    HIDDEN_LAYERS = [32]
    ENV_NAME = 'BipedalWalker-v3'  # 使用更简单的环境

# ==================== 文件路径 ====================
BEST_MODEL_PATH = "best_model.npy"
STATS_PATH = "training_stats.npy"
CHECKPOINT_PREFIX = "best_model_gen"

# ==================== 显示配置 ====================
VERBOSE = True  # 是否显示详细信息
SHOW_PROGRESS = True  # 是否显示评估进度


def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"环境: {ENV_NAME}")
    print(f"种群大小: {POPULATION_SIZE}")
    print(f"训练代数: {GENERATIONS}")
    print(f"变异率: {MUTATION_RATE}")
    print(f"交叉率: {CROSSOVER_RATE}")
    print(f"精英比例: {ELITE_RATIO}")
    print(f"神经网络结构: 输入 → {' → '.join(map(str, HIDDEN_LAYERS))} → 输出")
    print(f"每集最大步数: {MAX_STEPS}")
    if QUICK_TEST:
        print("⚠️  快速测试模式已启用")
    print("=" * 60)
    print()


if __name__ == "__main__":
    print_config()

