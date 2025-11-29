"""
配置文件
集中管理所有训练参数
"""

# ==================== 环境配置 ====================
ENV_NAME = 'CartPole-v1'  # 环境名称: 'CartPole-v1', 'BipedalWalker-v3', 'Hopper-v4'
FALLBACK_ENV = 'CartPole-v1'  # 备选环境

# ==================== 遗传算法参数 ====================
# CartPole专用配置
POPULATION_SIZE = 50        # CartPole种群大小
GENERATIONS = 100           # CartPole代数
MUTATION_RATE = 0.15        # CartPole变异率（较小）
MUTATION_SCALE = 0.2        # CartPole变异幅度（较小）
CROSSOVER_RATE = 0.8        # 交叉率
ELITE_RATIO = 0.1           # 精英比例（保留更多好的）
TOURNAMENT_SIZE = 3         # 锦标赛大小

# ==================== 神经网络参数 ====================
HIDDEN_LAYERS = [16, 16]  # CartPole网络（简单任务用小网络）

# ==================== 训练参数 ====================
MAX_STEPS = 500             # CartPole最大步数
RANDOM_SEED = None          # 随机种子（None=随机，提高成功率）
USE_FIXED_TERRAIN = False   # CartPole不需要
TERRAIN_SEEDS = None        # CartPole不需要
EVAL_EPISODES = 1           # 单次评估
RENDER_BEST = False         # 是否渲染最佳个体（改为False加快训练）
RENDER_FREQUENCY = 1        # 每隔几代渲染一次（设为1表示每代都渲染）
SAVE_FREQUENCY = 50         # 每隔50代保存一次模型
VIDEO_FREQUENCY = 999999    # 不定期录视频（只在NEW RECORD时录）

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

