"""
训练程序（带视频录制功能）
定期保存最优个体的演示视频，方便对比不同代数的训练效果
"""

import gymnasium as gym
import numpy as np
import time
import os
import sys
from datetime import datetime
from neural_network import NeuralNetwork, save_network
from genetic_algorithm import GeneticAlgorithm, AdaptiveGeneticAlgorithm
import config


def evaluate_individual(params, network, env, max_steps, terrain_seeds=None):
    """
    评估单个个体的适应度
    
    Args:
        terrain_seeds: 如果提供，使用固定地形seeds评估；否则随机评估
    
    Returns:
        avg_reward: 平均奖励
        best_seed: 表现最好的那次的seed（用于录视频）
    """
    network.set_params(params)
    
    if terrain_seeds is not None:
        # 固定地形模式：在多个固定地形上评估，取平均
        episode_rewards = []
        for seed in terrain_seeds:
            observation, info = env.reset(seed=seed)
            episode_reward = 0.0
            
            for step in range(max_steps):
                action = network.predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        # 返回平均奖励和第一个seed（固定地形时用于录视频）
        return np.mean(episode_rewards), terrain_seeds[0]
    else:
        # 随机地形模式（兼容旧代码）
        seed = np.random.randint(0, 1000000)
        observation, info = env.reset(seed=seed)
        episode_reward = 0.0
        
        for step in range(max_steps):
            action = network.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        return episode_reward, seed


def save_video_of_best(params, network, env_name, generation, max_steps, seed=None, video_dir="videos"):
    """
    保存最优个体的视频
    
    Args:
        params: 神经网络参数
        network: 神经网络实例
        env_name: 环境名称
        generation: 当前代数
        max_steps: 最大步数
        seed: 环境随机种子（固定地形模式下必须提供）
        video_dir: 视频保存目录
    """
    # 创建视频目录
    os.makedirs(video_dir, exist_ok=True)
    
    network.set_params(params)
    
    # 使用提供的seed（固定地形）
    if seed is None:
        seed = 42  # 默认seed
    
    best_seed = seed
    
    # 使用最好的seed录制视频
    try:
        video_path = os.path.join(video_dir, f"gen_{generation:03d}")
        env = gym.make(env_name, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(
            env, 
            video_path,
            episode_trigger=lambda x: True,
            name_prefix=f"best_gen{generation}"
        )
        
        observation, info = env.reset(seed=best_seed)
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            action = network.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        env.close()
        
        print(f"OK (terrain seed={best_seed}, reward: {total_reward:.1f}, steps: {steps})", end=' ')
        sys.stdout.flush()
        return total_reward
        
    except Exception as e:
        print(f"Failed: {e}", end=' ')
        sys.stdout.flush()
        return 0.0


def main():
    """主函数"""
    # 显示配置
    config.print_config()
    
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"视频保存频率: 每 {config.VIDEO_FREQUENCY} 代\n")
    
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
    
    # 创建标准遗传算法（固定变异率，持续探索）
    ga = GeneticAlgorithm(
        population_size=config.POPULATION_SIZE,
        param_count=network.get_param_count(),
        mutation_rate=config.MUTATION_RATE,
        mutation_scale=config.MUTATION_SCALE,
        crossover_rate=config.CROSSOVER_RATE,
        elite_ratio=config.ELITE_RATIO,
        tournament_size=config.TOURNAMENT_SIZE
    )
    print(f"使用标准遗传算法（固定高变异率：{config.MUTATION_RATE}）")
    
    # 创建训练环境（不渲染，速度快）
    env_train = gym.make(env_name)
    
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
    print("开始训练（无实时渲染，速度更快）")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    for generation in range(config.GENERATIONS):
        gen_start = time.time()
        
        print(f"\n[Gen {generation + 1}/{config.GENERATIONS}]", end=' ')
        sys.stdout.flush()
        
        # 评估种群
        # 如果使用固定地形，传入固定的地形seeds；否则为None（随机）
        terrain_seeds = config.TERRAIN_SEEDS if config.USE_FIXED_TERRAIN else None
        
        fitness_scores = []
        individual_seeds = []  # 记录每个个体评估时的seed
        
        for i, individual in enumerate(ga.population):
            avg_fitness, eval_seed = evaluate_individual(
                individual, network, env_train, config.MAX_STEPS, terrain_seeds
            )
            fitness_scores.append(avg_fitness)
            individual_seeds.append(eval_seed)  # 保存每个个体的seed
            
            if config.SHOW_PROGRESS and (i + 1) % 10 == 0:
                print(f"Eval: {i + 1}/{config.POPULATION_SIZE}...", end=' ')
                sys.stdout.flush()
        
        ga.fitness_scores = np.array(fitness_scores)
        
        # 统计
        stats = ga.get_statistics()
        best_idx = np.argmax(fitness_scores)
        best_individual = ga.population[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        # 录视频用的seed：使用最佳个体评估时的seed
        video_seed = individual_seeds[best_idx]
        
        is_new_record = False
        
        if best_fitness > best_ever_fitness:
            is_new_record = True
            best_ever_fitness = best_fitness
            best_ever_params = best_individual.copy()
            print(f"NEW RECORD! ", end='')
            sys.stdout.flush()
        
        # 记录
        stats_history['generation'].append(generation + 1)
        stats_history['best_fitness'].append(stats['best'])
        stats_history['mean_fitness'].append(stats['mean'])
        stats_history['std_fitness'].append(stats['std'])
        
        # 显示统计信息
        print(f"Best: {stats['best']:.2f}, Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}", end=' ')
        sys.stdout.flush()
        
        # 如果是新记录，立即录制视频（使用固定地形）
        if is_new_record and best_ever_params is not None:
            print(f"| Recording video...", end=' ')
            sys.stdout.flush()
            save_video_of_best(
                best_ever_params,
                network, 
                env_name, 
                generation + 1, 
                config.MAX_STEPS,
                seed=video_seed  # 使用固定地形的seed
            )
        # 定期保存视频（即使不是新记录）
        elif ((generation + 1) % config.VIDEO_FREQUENCY == 0) and best_ever_params is not None:
            print(f"| Recording video...", end=' ')
            sys.stdout.flush()
            save_video_of_best(
                best_ever_params,
                network, 
                env_name, 
                generation + 1, 
                config.MAX_STEPS,
                seed=video_seed
            )
        
        # 保存检查点
        if (generation + 1) % config.SAVE_FREQUENCY == 0:
            print(f"| Saving checkpoint...", end=' ')
            sys.stdout.flush()
            network.set_params(best_ever_params)
            save_network(network, f"{config.CHECKPOINT_PREFIX}{generation+1}.npy")
            np.save(config.STATS_PATH, stats_history)
            print(f"Done", end=' ')
            sys.stdout.flush()
        
        # 进化
        if generation < config.GENERATIONS - 1:
            ga.evolve()
        
        gen_time = time.time() - gen_start
        print(f"| Time: {gen_time:.1f}s")
        sys.stdout.flush()
    
    # 完成
    total_time = time.time() - start_time
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"总用时: {total_time/60:.1f}分钟")
    print(f"最佳适应度: {best_ever_fitness:.2f}\n")
    
    # 保存最终模型
    network.set_params(best_ever_params)
    save_network(network, config.BEST_MODEL_PATH)
    np.save(config.STATS_PATH, stats_history)
    
    # 保存最终视频（使用固定地形）
    print("📹 录制最终最佳个体视频...")
    final_seed = config.TERRAIN_SEEDS[0] if config.USE_FIXED_TERRAIN else 42
    save_video_of_best(
        best_ever_params, 
        network, 
        env_name, 
        config.GENERATIONS, 
        config.MAX_STEPS,
        seed=final_seed
    )
    
    env_train.close()
    
    print(f"\n模型已保存为: {config.BEST_MODEL_PATH}")
    print(f"视频已保存到: videos/ 目录")
    print(f"训练统计已保存为: {config.STATS_PATH}")
    print("\n使用 python visualize.py 查看训练曲线和最终效果")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

