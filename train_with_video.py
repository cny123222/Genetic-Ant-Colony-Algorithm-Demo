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


def save_video_of_best(params, network, env_name, generation, max_steps, video_dir="videos", num_trials=3):
    """
    保存最优个体的视频（录制多次，保存最好的一次）
    
    Args:
        params: 神经网络参数
        network: 神经网络实例
        env_name: 环境名称
        generation: 当前代数
        max_steps: 最大步数
        video_dir: 视频保存目录
        num_trials: 尝试次数（取最好的）
    """
    # 创建视频目录
    os.makedirs(video_dir, exist_ok=True)
    
    network.set_params(params)
    
    # 先运行多次找到最好的seed
    best_reward = -np.inf
    best_seed = 0
    
    for trial in range(num_trials):
        env_test = gym.make(env_name)
        observation, info = env_test.reset(seed=trial)
        trial_reward = 0.0
        
        for step in range(max_steps):
            action = network.predict(observation)
            observation, reward, terminated, truncated, info = env_test.step(action)
            trial_reward += reward
            if terminated or truncated:
                break
        
        env_test.close()
        
        if trial_reward > best_reward:
            best_reward = trial_reward
            best_seed = trial
    
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
        
        print(f"OK (best of {num_trials} trials, reward: {total_reward:.1f}, steps: {steps})", end=' ')
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
        fitness_scores = []
        for i, individual in enumerate(ga.population):
            fitness = evaluate_individual(individual, network, env_train, config.MAX_STEPS)
            fitness_scores.append(fitness)
            
            if config.SHOW_PROGRESS and (i + 1) % 10 == 0:
                print(f"Eval: {i + 1}/{config.POPULATION_SIZE}...", end=' ')
                sys.stdout.flush()
        
        ga.fitness_scores = np.array(fitness_scores)
        
        # 统计
        stats = ga.get_statistics()
        best_individual, best_fitness = ga.get_best_individual()
        
        if best_fitness > best_ever_fitness:
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
        
        # 定期保存视频（使用历史最佳个体）
        if ((generation + 1) % config.VIDEO_FREQUENCY == 0 or generation == 0) and best_ever_params is not None:
            print(f"| Recording video...", end=' ')
            sys.stdout.flush()
            save_video_of_best(
                best_ever_params,  # 使用历史最佳
                network, 
                env_name, 
                generation + 1, 
                config.MAX_STEPS
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
    
    # 保存最终视频
    print("📹 录制最终最佳个体视频...")
    save_video_of_best(
        best_ever_params, 
        network, 
        env_name, 
        config.GENERATIONS, 
        config.MAX_STEPS
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

