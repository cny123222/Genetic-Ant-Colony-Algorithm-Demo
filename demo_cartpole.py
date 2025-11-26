"""
演示脚本：使用遗传算法玩 CartPole
这是一个更简单的环境，用于快速验证算法是否正常工作
"""

import gymnasium as gym
import numpy as np
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm


def evaluate_cartpole(params, network, env, episodes=3):
    """评估CartPole性能"""
    network.set_params(params)
    total_reward = 0
    
    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0
        
        for _ in range(500):  # CartPole最多500步
            action = network.predict(observation)
            # CartPole需要离散动作 (0或1)
            action = 1 if action[0] > 0 else 0
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / episodes


def demo_cartpole():
    """CartPole演示"""
    print("=" * 60)
    print("遗传算法演示 - CartPole 平衡杆")
    print("=" * 60)
    print()
    
    # 创建环境
    env_train = gym.make('CartPole-v1')
    env_render = gym.make('CartPole-v1', render_mode='human')
    
    obs_dim = env_train.observation_space.shape[0]  # 4
    act_dim = 1  # 输出一个值，然后转换为0或1
    
    print(f"环境: CartPole-v1")
    print(f"目标: 保持杆子平衡尽可能长时间（最大500步）")
    print(f"观察空间: {obs_dim}维")
    print(f"动作空间: 离散 (左/右)")
    print()
    
    # 创建神经网络（简单结构）
    network = NeuralNetwork(
        input_size=obs_dim,
        hidden_sizes=[16],  # 单层16个神经元
        output_size=act_dim
    )
    
    print(f"神经网络: {obs_dim} → 16 → {act_dim}")
    print(f"参数数量: {network.get_param_count()}")
    print()
    
    # 创建遗传算法
    ga = GeneticAlgorithm(
        population_size=30,
        param_count=network.get_param_count(),
        mutation_rate=0.2,
        mutation_scale=0.5,
        crossover_rate=0.7,
        elite_ratio=0.1
    )
    
    print("开始训练...")
    print("-" * 60)
    
    best_ever_fitness = 0
    
    for generation in range(50):
        # 评估种群
        fitness_scores = []
        for individual in ga.population:
            fitness = evaluate_cartpole(individual, network, env_train, episodes=3)
            fitness_scores.append(fitness)
        
        ga.fitness_scores = np.array(fitness_scores)
        
        # 获取最佳个体
        best_individual, best_fitness = ga.get_best_individual()
        stats = ga.get_statistics()
        
        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
        
        print(f"第 {generation + 1:2d} 代 | "
              f"最佳: {stats['best']:6.1f} | "
              f"平均: {stats['mean']:6.1f} | "
              f"历史最佳: {best_ever_fitness:6.1f}")
        
        # 每10代展示一次
        if (generation + 1) % 10 == 0:
            print(f"  → 展示当前最佳个体...")
            demo_reward = evaluate_cartpole(best_individual, network, env_render, episodes=1)
            print(f"     展示得分: {demo_reward:.1f}")
        
        # 进化
        if generation < 49:
            ga.evolve()
        
        # 如果已经解决问题（平均475+），提前结束
        if stats['mean'] >= 475:
            print(f"\n🎉 问题已解决！平均得分 {stats['mean']:.1f} >= 475")
            break
    
    print()
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最终最佳得分: {best_ever_fitness:.1f}")
    print()
    print("展示最终最佳个体（5次）...")
    
    for i in range(5):
        reward = evaluate_cartpole(best_individual, network, env_render, episodes=1)
        print(f"  测试 {i+1}: {reward:.1f}")
    
    env_train.close()
    env_render.close()
    
    print()
    print("演示完成！")
    print("CartPole是一个简单的问题，通常在20-30代内就能解决。")
    print("如果这个演示运行良好，说明算法实现正确，可以尝试更复杂的环境。")


if __name__ == "__main__":
    try:
        demo_cartpole()
    except KeyboardInterrupt:
        print("\n\n演示被中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

