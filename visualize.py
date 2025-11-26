"""
可视化脚本
加载训练好的模型并展示结果
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from neural_network import load_network
import sys
import os

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def visualize_model(model_path="best_model.npy", num_episodes=5, max_steps=1000):
    """
    加载并可视化训练好的模型
    
    Args:
        model_path: 模型文件路径
        num_episodes: 运行的episode数量
        max_steps: 每个episode的最大步数
    """
    print("=" * 60)
    print("遗传算法训练结果可视化")
    print("=" * 60)
    print()
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"✗ 错误: 找不到模型文件 {model_path}")
        print("请先运行 train_humanoid.py 进行训练")
        return
    
    # 加载模型
    print(f"加载模型: {model_path}")
    try:
        network = load_network(model_path)
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return
    print()
    
    # 创建环境
    print("初始化环境...")
    try:
        # 尝试 Humanoid-v4
        env = gym.make('Humanoid-v4', render_mode='human')
        env_name = "Humanoid-v4"
    except:
        try:
            # 备选: BipedalWalker-v3
            env = gym.make('BipedalWalker-v3', render_mode='human')
            env_name = "BipedalWalker-v3"
        except Exception as e:
            print(f"✗ 无法创建环境: {e}")
            return
    
    print(f"✓ 环境: {env_name}")
    print()
    
    # 运行多个episodes
    print(f"运行 {num_episodes} 个episodes...")
    print("按 Ctrl+C 可随时中断")
    print("-" * 60)
    
    episode_rewards = []
    episode_steps = []
    
    try:
        for episode in range(num_episodes):
            observation, info = env.reset()
            total_reward = 0.0
            steps = 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            for step in range(max_steps):
                # 使用神经网络预测动作
                action = network.predict(observation)
                
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # 检查是否结束
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            print(f"  总奖励: {total_reward:.2f}")
            print(f"  存活步数: {steps}/{max_steps}")
            
    except KeyboardInterrupt:
        print("\n\n已中断")
    
    # 关闭环境
    env.close()
    
    # 显示统计信息
    if episode_rewards:
        print()
        print("=" * 60)
        print("统计信息")
        print("=" * 60)
        print(f"Episodes数量: {len(episode_rewards)}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"标准差: {np.std(episode_rewards):.2f}")
        print(f"最佳奖励: {np.max(episode_rewards):.2f}")
        print(f"最差奖励: {np.min(episode_rewards):.2f}")
        print(f"平均存活步数: {np.mean(episode_steps):.1f}")
        print()


def plot_training_stats(stats_path="training_stats.npy"):
    """
    绘制训练统计图表
    
    Args:
        stats_path: 统计数据文件路径
    """
    if not os.path.exists(stats_path):
        print(f"✗ 找不到统计文件 {stats_path}")
        return
    
    print(f"加载训练统计: {stats_path}")
    stats = np.load(stats_path, allow_pickle=True).item()
    
    generations = stats['generation']
    best_fitness = stats['best_fitness']
    mean_fitness = stats['mean_fitness']
    std_fitness = stats['std_fitness']
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制适应度曲线
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, mean_fitness, 'g--', label='Mean Fitness', linewidth=1.5)
    plt.fill_between(
        generations,
        np.array(mean_fitness) - np.array(std_fitness),
        np.array(mean_fitness) + np.array(std_fitness),
        alpha=0.3,
        color='green',
        label='Std Range'
    )
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Training Progress - Fitness Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制改进速度
    plt.subplot(1, 2, 2)
    if len(best_fitness) > 1:
        improvement = np.diff(best_fitness)
        plt.plot(generations[1:], improvement, 'r-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Improvement', fontsize=12)
        plt.title('Fitness Improvement per Generation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Training chart saved as training_progress.png")
    
    plt.show()
    
    # 显示统计摘要
    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total Generations: {len(generations)}")
    print(f"Initial Best Fitness: {best_fitness[0]:.2f}")
    print(f"Final Best Fitness: {best_fitness[-1]:.2f}")
    print(f"Total Improvement: {best_fitness[-1] - best_fitness[0]:.2f} ({(best_fitness[-1]/best_fitness[0] - 1)*100 if best_fitness[0] > 0 else 0:.1f}%)")
    print(f"Final Mean Fitness: {mean_fitness[-1]:.2f}")
    print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化遗传算法训练结果')
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.npy',
        help='模型文件路径'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='运行的episode数量'
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='仅绘制训练曲线，不运行模型'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='不绘制训练曲线，仅运行模型'
    )
    
    args = parser.parse_args()
    
    if args.plot_only:
        # 仅绘制训练曲线
        plot_training_stats()
    elif args.no_plot:
        # 仅运行模型
        visualize_model(args.model, args.episodes)
    else:
        # 先绘制训练曲线
        if os.path.exists("training_stats.npy"):
            plot_training_stats()
            print()
            input("按 Enter 键继续展示模型...")
            print()
        
        # 然后运行模型
        visualize_model(args.model, args.episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已退出")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()

