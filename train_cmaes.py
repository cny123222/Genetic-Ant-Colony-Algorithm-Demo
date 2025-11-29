"""
CMA-ES训练脚本
使用协方差矩阵自适应进化策略
比标准GA在高维空间更高效
"""

import numpy as np
import gymnasium as gym
import cma
import os
import time
from datetime import datetime
from neural_network import NeuralNetwork, save_network, create_random_params
import config


def evaluate_individual(network, env_name, max_steps, seed=None):
    """评估单个个体的适应度"""
    env = gym.make(env_name)
    
    if seed is not None:
        observation, _ = env.reset(seed=seed)
    else:
        observation, _ = env.reset()
    
    total_reward = 0
    steps = 0
    
    for _ in range(max_steps):
        action = network.predict(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    env.close()
    return total_reward


def save_video_of_best(params, network, env_name, generation, max_steps):
    """保存最佳个体的视频"""
    from gymnasium.wrappers import RecordVideo
    
    # 设置参数
    network.set_params(params)
    
    # 创建视频保存目录
    video_folder = f"videos_cmaes/gen_{generation:03d}"
    os.makedirs(video_folder, exist_ok=True)
    
    # 创建环境并录制
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_folder,
        name_prefix=f"best_gen{generation}",
        episode_trigger=lambda x: True
    )
    
    observation, _ = env.reset(seed=config.RANDOM_SEED if config.RANDOM_SEED else None)
    total_reward = 0
    
    for _ in range(max_steps):
        action = network.predict(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.close()
    return total_reward


def main():
    """主训练函数"""
    # 设置随机种子
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)
    
    # 创建环境以获取维度
    env = gym.make(config.ENV_NAME)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    env.close()
    
    # 创建神经网络
    network = NeuralNetwork(input_size, config.HIDDEN_LAYERS, output_size)
    param_count = network.param_count
    
    print("=" * 80)
    print("CMA-ES训练开始")
    print("=" * 80)
    print(f"环境: {config.ENV_NAME}")
    print(f"网络结构: {input_size} → {' → '.join(map(str, config.HIDDEN_LAYERS))} → {output_size}")
    print(f"参数总数: {param_count:,}")
    print(f"种群大小: 由CMA-ES自动确定（默认 4 + 3*ln(N) ≈ {4 + int(3 * np.log(param_count))}）")
    print(f"最大迭代数: {config.GENERATIONS}")
    print(f"随机种子: {config.RANDOM_SEED}")
    print("=" * 80)
    print()
    
    # 初始化CMA-ES
    initial_params = create_random_params(param_count, scale=0.1)
    sigma0 = 0.5  # 初始步长
    
    # CMA-ES选项
    cma_options = {
        'maxiter': config.GENERATIONS,
        'popsize': 100,  # 固定种群大小为100，与之前GA保持一致
        'verb_disp': 1,  # 每代显示信息
        'verb_log': 0,   # 不保存日志文件
        'seed': config.RANDOM_SEED if config.RANDOM_SEED else None
    }
    
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, cma_options)
    
    # 训练统计
    best_ever_fitness = float('-inf')
    best_ever_params = None
    training_start = time.time()
    
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        iteration = 0
        while not es.stop():
            iteration += 1
            gen_start = time.time()
            
            # 生成候选解
            solutions = es.ask()
            
            # 评估所有候选解
            fitness_scores = []
            for i, params in enumerate(solutions):
                network.set_params(params)
                fitness = evaluate_individual(
                    network, 
                    config.ENV_NAME, 
                    config.MAX_STEPS,
                    seed=config.RANDOM_SEED if config.RANDOM_SEED else None
                )
                fitness_scores.append(fitness)
                
                if config.SHOW_PROGRESS and (i + 1) % 10 == 0:
                    print(f"Eval: {i+1}/{len(solutions)}...", end=" ", flush=True)
            
            if config.SHOW_PROGRESS:
                print()  # 换行
            
            # 告诉CMA-ES评估结果（注意：CMA-ES最小化，所以取负值）
            es.tell(solutions, [-f for f in fitness_scores])
            
            # 统计信息
            best_fitness = max(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            best_params = solutions[best_idx]
            mean_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            
            # 更新全局最优
            new_record = False
            if best_fitness > best_ever_fitness:
                best_ever_fitness = best_fitness
                best_ever_params = best_params.copy()
                new_record = True
            
            # 显示进度
            gen_time = time.time() - gen_start
            if new_record:
                print(f"[Gen {iteration}/{config.GENERATIONS}] ⭐ NEW RECORD! Best: {best_fitness:.2f}, Mean: {mean_fitness:.2f}, Std: {std_fitness:.2f} | Time: {gen_time:.1f}s")
                
                # 保存视频
                if best_ever_params is not None:
                    print("Recording video...", end=" ", flush=True)
                    video_reward = save_video_of_best(
                        best_ever_params,
                        network,
                        config.ENV_NAME,
                        iteration,
                        config.MAX_STEPS
                    )
                    print(f"OK (reward: {video_reward:.1f})")
            else:
                print(f"[Gen {iteration}/{config.GENERATIONS}] Best: {best_fitness:.2f}, Mean: {mean_fitness:.2f}, Std: {std_fitness:.2f} | Time: {gen_time:.1f}s")
            
            # 定期保存模型
            if iteration % config.SAVE_FREQUENCY == 0 and best_ever_params is not None:
                checkpoint_path = f"models_cmaes/best_model_gen{iteration}.npy"
                os.makedirs("models_cmaes", exist_ok=True)
                network.set_params(best_ever_params)
                save_network(network, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            print()
    
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    
    # 训练结束
    training_time = time.time() - training_start
    
    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"总用时: {training_time/3600:.2f} 小时 ({training_time/60:.1f} 分钟)")
    print(f"最终最佳适应度: {best_ever_fitness:.2f}")
    print(f"平均每代用时: {training_time/iteration:.1f} 秒")
    print("=" * 80)
    
    # 保存最终模型
    if best_ever_params is not None:
        network.set_params(best_ever_params)
        final_model_path = "best_model_cmaes.npy"
        save_network(network, final_model_path)
        print(f"\n✅ 最终模型已保存: {final_model_path}")
        
        # 保存最终视频
        print("\n录制最终演示视频...", end=" ", flush=True)
        final_reward = save_video_of_best(
            best_ever_params,
            network,
            config.ENV_NAME,
            iteration,
            config.MAX_STEPS
        )
        print(f"完成！(reward: {final_reward:.1f})")
    
    print("\n🎉 CMA-ES训练全部完成！")


if __name__ == "__main__":
    main()

