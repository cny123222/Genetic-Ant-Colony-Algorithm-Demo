"""
简化版训练程序（使用config.py配置）
更容易配置和使用
"""

import gymnasium as gym
import numpy as np
import time
from datetime import datetime
from neural_network import NeuralNetwork, save_network
from genetic_algorithm import GeneticAlgorithm
import config


def evaluate_individual(params, network, env, max_steps):
    """评估单个个体的适应度"""
    network.set_params(params)
    observation, info = env.reset()
    total_reward = 0.0
    
    for step in range(max_steps):
        action = network.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward


def main():
    """主函数"""
    # 显示配置
    config.print_config()
    
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 初始化环境
    print("初始化环境...")
    try:
        env_test = gym.make(config.ENV_NAME)
        env_name = config.ENV_NAME
        print(f"✓ 成功加载 {env_name}")
    except Exception as e:
        print(f"✗ 无法加载 {config.ENV_NAME}: {e}")
        print(f"尝试备选环境 {config.FALLBACK_ENV}...")
        try:
            env_test = gym.make(config.FALLBACK_ENV)
            env_name = config.FALLBACK_ENV
            print(f"✓ 成功加载 {env_name}")
        except Exception as e2:
            print(f"✗ 无法加载备选环境: {e2}")
            return
    
    obs_dim = env_test.observation_space.shape[0]
    act_dim = env_test.action_space.shape[0]
    env_test.close()
    
    print(f"观察空间: {obs_dim}维, 动作空间: {act_dim}维\n")
    
    # 创建神经网络
    network = NeuralNetwork(obs_dim, config.HIDDEN_LAYERS, act_dim)
    print(f"网络参数数量: {network.get_param_count()}\n")
    
    # 创建遗传算法
    ga = GeneticAlgorithm(
        population_size=config.POPULATION_SIZE,
        param_count=network.get_param_count(),
        mutation_rate=config.MUTATION_RATE,
        mutation_scale=config.MUTATION_SCALE,
        crossover_rate=config.CROSSOVER_RATE,
        elite_ratio=config.ELITE_RATIO,
        tournament_size=config.TOURNAMENT_SIZE
    )
    
    # 创建环境
    env_train = gym.make(env_name)
    env_render = None
    if config.RENDER_BEST:
        try:
            env_render = gym.make(env_name, render_mode='human')
        except:
            pass
    
    # 训练统计
    stats_history = {
        'generation': [],
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': []
    }
    
    best_ever_fitness = -np.inf
    best_ever_params = None
    
    print("=" * 60)
    print("开始训练")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    for generation in range(config.GENERATIONS):
        gen_start = time.time()
        
        print(f"代 {generation + 1}/{config.GENERATIONS}")
        
        # 评估种群
        fitness_scores = []
        for i, individual in enumerate(ga.population):
            fitness = evaluate_individual(individual, network, env_train, config.MAX_STEPS)
            fitness_scores.append(fitness)
            
            if config.SHOW_PROGRESS and (i + 1) % 10 == 0:
                print(f"  评估进度: {i + 1}/{config.POPULATION_SIZE}", end='\r')
        
        if config.SHOW_PROGRESS:
            print()
        
        ga.fitness_scores = np.array(fitness_scores)
        
        # 统计
        stats = ga.get_statistics()
        best_individual, best_fitness = ga.get_best_individual()
        
        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_params = best_individual.copy()
            print(f"  🎉 新纪录！适应度: {best_fitness:.2f}")
        
        # 记录
        stats_history['generation'].append(generation + 1)
        stats_history['best_fitness'].append(stats['best'])
        stats_history['mean_fitness'].append(stats['mean'])
        stats_history['std_fitness'].append(stats['std'])
        
        print(f"  最佳: {stats['best']:.2f}, 平均: {stats['mean']:.2f}, "
              f"标准差: {stats['std']:.2f}")
        
        # 渲染
        if config.RENDER_BEST and env_render and (generation % config.RENDER_FREQUENCY == 0):
            evaluate_individual(best_individual, network, env_render, config.MAX_STEPS)
        
        # 保存检查点
        if (generation + 1) % config.SAVE_FREQUENCY == 0:
            network.set_params(best_ever_params)
            save_network(network, f"{config.CHECKPOINT_PREFIX}{generation+1}.npy")
            np.save(config.STATS_PATH, stats_history)
        
        # 进化
        if generation < config.GENERATIONS - 1:
            ga.evolve()
        
        print(f"  用时: {time.time() - gen_start:.1f}秒\n")
    
    # 完成
    total_time = time.time() - start_time
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"总用时: {total_time/60:.1f}分钟")
    print(f"最佳适应度: {best_ever_fitness:.2f}\n")
    
    # 保存
    network.set_params(best_ever_params)
    save_network(network, config.BEST_MODEL_PATH)
    np.save(config.STATS_PATH, stats_history)
    
    # 最终展示
    if config.RENDER_BEST and env_render:
        print("\n最终展示（3个episodes）...")
        for i in range(3):
            reward = evaluate_individual(best_ever_params, network, env_render, config.MAX_STEPS)
            print(f"Episode {i+1}: {reward:.2f}")
    
    env_train.close()
    if env_render:
        env_render.close()
    
    print(f"\n模型已保存为: {config.BEST_MODEL_PATH}")
    print("使用 python visualize.py 查看结果")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

