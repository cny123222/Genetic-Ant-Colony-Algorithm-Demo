"""
主训练程序
使用遗传算法训练3D人形机器人行走
"""

import gymnasium as gym
import numpy as np
import time
from datetime import datetime
from neural_network import NeuralNetwork, save_network
from genetic_algorithm import GeneticAlgorithm

# ==================== 训练参数配置 ====================
POPULATION_SIZE = 50        # 种群大小
GENERATIONS = 100           # 训练代数
MUTATION_RATE = 0.15        # 变异率
MUTATION_SCALE = 0.3        # 变异幅度
CROSSOVER_RATE = 0.8        # 交叉率
ELITE_RATIO = 0.1           # 精英比例
TOURNAMENT_SIZE = 3         # 锦标赛大小

MAX_STEPS = 1000           # 每个episode的最大步数
RENDER_BEST = True         # 是否渲染最佳个体
RENDER_FREQUENCY = 1       # 每隔几代渲染一次

HIDDEN_LAYERS = [64, 32]   # 神经网络隐藏层结构

SAVE_FREQUENCY = 10        # 每隔几代保存一次模型
# ====================================================


def evaluate_individual(params, network, env, render=False, max_steps=MAX_STEPS):
    """
    评估单个个体的适应度
    
    Args:
        params: 神经网络参数向量
        network: NeuralNetwork 实例
        env: gym 环境
        render: 是否渲染
        max_steps: 最大步数
        
    Returns:
        总奖励（适应度分数）
    """
    network.set_params(params)
    
    observation, info = env.reset()
    total_reward = 0.0
    
    for step in range(max_steps):
        # 使用神经网络预测动作
        action = network.predict(observation)
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 检查是否结束
        if terminated or truncated:
            break
    
    return total_reward


def train():
    """主训练函数"""
    print("=" * 60)
    print("遗传算法训练3D人形机器人行走")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 创建环境（用于获取维度信息）
    print("初始化环境...")
    try:
        env_test = gym.make('Humanoid-v4')
        print("✓ 成功加载 Humanoid-v4 环境")
    except Exception as e:
        print(f"✗ 无法加载 Humanoid-v4: {e}")
        print("尝试使用备选环境 BipedalWalker-v3...")
        try:
            env_test = gym.make('BipedalWalker-v3')
            print("✓ 成功加载 BipedalWalker-v3 环境")
        except Exception as e2:
            print(f"✗ 无法加载备选环境: {e2}")
            print("请确保已安装 gymnasium 和相关依赖")
            return
    
    obs_dim = env_test.observation_space.shape[0]
    act_dim = env_test.action_space.shape[0]
    env_test.close()
    
    print(f"观察空间维度: {obs_dim}")
    print(f"动作空间维度: {act_dim}")
    print()
    
    # 创建神经网络
    print("创建神经网络控制器...")
    network = NeuralNetwork(
        input_size=obs_dim,
        hidden_sizes=HIDDEN_LAYERS,
        output_size=act_dim
    )
    param_count = network.get_param_count()
    print(f"网络结构: {obs_dim} → {' → '.join(map(str, HIDDEN_LAYERS))} → {act_dim}")
    print(f"总参数数量: {param_count}")
    print()
    
    # 创建遗传算法
    print("初始化遗传算法...")
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        param_count=param_count,
        mutation_rate=MUTATION_RATE,
        mutation_scale=MUTATION_SCALE,
        crossover_rate=CROSSOVER_RATE,
        elite_ratio=ELITE_RATIO,
        tournament_size=TOURNAMENT_SIZE
    )
    print(f"种群大小: {POPULATION_SIZE}")
    print(f"精英数量: {ga.elite_count}")
    print()
    
    # 创建训练环境（不渲染）
    env_train = gym.make('Humanoid-v4') if 'Humanoid' in str(env_test) else gym.make('BipedalWalker-v3')
    
    # 创建渲染环境（用于展示最佳个体）
    env_render = None
    if RENDER_BEST:
        try:
            env_render = gym.make('Humanoid-v4', render_mode='human') if 'Humanoid' in str(env_test) else gym.make('BipedalWalker-v3', render_mode='human')
            print("✓ 渲染模式已启用")
        except:
            print("✗ 无法启用渲染模式，将仅显示统计数据")
            RENDER_BEST = False
    print()
    
    # 统计信息存储
    stats_history = {
        'generation': [],
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': []
    }
    
    best_ever_fitness = -np.inf
    best_ever_params = None
    
    # 开始训练
    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    
    start_time = time.time()
    
    for generation in range(GENERATIONS):
        gen_start_time = time.time()
        
        print(f"\n代 {generation + 1}/{GENERATIONS}")
        print("-" * 60)
        
        # 评估所有个体
        print("评估种群...")
        fitness_scores = []
        for i, individual in enumerate(ga.population):
            fitness = evaluate_individual(individual, network, env_train)
            fitness_scores.append(fitness)
            
            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == POPULATION_SIZE:
                print(f"  进度: {i + 1}/{POPULATION_SIZE}", end='\r')
        
        ga.fitness_scores = np.array(fitness_scores)
        print()
        
        # 获取统计信息
        stats = ga.get_statistics()
        best_individual, best_fitness = ga.get_best_individual()
        
        # 更新历史最佳
        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_params = best_individual.copy()
            print(f"🎉 发现新的最佳个体！适应度: {best_fitness:.2f}")
        
        # 记录统计信息
        stats_history['generation'].append(generation + 1)
        stats_history['best_fitness'].append(stats['best'])
        stats_history['mean_fitness'].append(stats['mean'])
        stats_history['std_fitness'].append(stats['std'])
        
        # 显示统计信息
        print(f"适应度 - 最佳: {stats['best']:.2f}, 平均: {stats['mean']:.2f}, "
              f"标准差: {stats['std']:.2f}, 最差: {stats['worst']:.2f}")
        
        # 渲染最佳个体
        if RENDER_BEST and env_render and (generation % RENDER_FREQUENCY == 0 or generation == GENERATIONS - 1):
            print("展示最佳个体表现...")
            eval_reward = evaluate_individual(
                best_individual, 
                network, 
                env_render, 
                render=True, 
                max_steps=MAX_STEPS
            )
            print(f"展示奖励: {eval_reward:.2f}")
        
        # 定期保存模型
        if (generation + 1) % SAVE_FREQUENCY == 0:
            network.set_params(best_ever_params)
            save_network(network, f"best_model_gen{generation+1}.npy")
            np.save("training_stats.npy", stats_history)
        
        # 进化到下一代
        if generation < GENERATIONS - 1:
            ga.evolve()
        
        gen_time = time.time() - gen_start_time
        print(f"本代用时: {gen_time:.1f}秒")
    
    # 训练结束
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"历史最佳适应度: {best_ever_fitness:.2f}")
    print()
    
    # 保存最终模型
    print("保存最终模型...")
    network.set_params(best_ever_params)
    save_network(network, "best_model.npy")
    np.save("training_stats.npy", stats_history)
    print()
    
    # 最终展示
    if RENDER_BEST and env_render:
        print("展示最终最佳个体...")
        print("按 Ctrl+C 可提前结束")
        try:
            for episode in range(3):
                print(f"\nEpisode {episode + 1}/3")
                final_reward = evaluate_individual(
                    best_ever_params,
                    network,
                    env_render,
                    render=True,
                    max_steps=MAX_STEPS
                )
                print(f"奖励: {final_reward:.2f}")
        except KeyboardInterrupt:
            print("\n已中断")
    
    # 清理
    env_train.close()
    if env_render:
        env_render.close()
    
    print()
    print("训练结果已保存:")
    print("  - best_model.npy: 最佳模型参数")
    print("  - training_stats.npy: 训练统计数据")
    print()
    print("使用 python visualize.py 来查看训练结果")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()

