"""
为已保存的模型生成演示视频
可以对比不同代数的训练效果
"""

import gymnasium as gym
import numpy as np
import os
import glob
from neural_network import load_network
import config


def generate_video_for_model(model_path, env_name, max_steps=500, video_dir="videos"):
    """
    为单个模型生成视频
    
    Args:
        model_path: 模型文件路径
        env_name: 环境名称
        max_steps: 最大步数
        video_dir: 视频保存目录
    """
    # 创建视频目录
    os.makedirs(video_dir, exist_ok=True)
    
    # 提取代数信息
    model_name = os.path.basename(model_path).replace('.npy', '')
    
    try:
        # 加载模型
        network = load_network(model_path)
        
        # 创建录制环境
        video_path = os.path.join(video_dir, model_name)
        env = gym.make(env_name, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(
            env, 
            video_path,
            episode_trigger=lambda x: True,
            name_prefix=model_name
        )
        
        # 运行一个episode
        observation, info = env.reset()
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
        
        print(f"✓ {model_name:20s} | 奖励: {total_reward:7.2f} | 步数: {steps:3d} | 视频: {video_path}/")
        return True
        
    except Exception as e:
        print(f"✗ {model_name:20s} | 错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("为已保存的模型生成演示视频")
    print("=" * 70)
    print()
    
    # 环境设置
    env_name = config.ENV_NAME
    print(f"环境: {env_name}")
    print(f"最大步数: {config.MAX_STEPS}")
    print()
    
    # 查找所有模型文件
    model_files = []
    
    # 查找检查点模型
    checkpoint_pattern = f"{config.CHECKPOINT_PREFIX}*.npy"
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    model_files.extend(checkpoints)
    
    # 添加最佳模型
    if os.path.exists(config.BEST_MODEL_PATH):
        model_files.append(config.BEST_MODEL_PATH)
    
    if not model_files:
        print("❌ 没有找到已保存的模型文件")
        print(f"   请先运行训练程序: python train_with_video.py")
        return
    
    print(f"找到 {len(model_files)} 个模型文件:\n")
    
    # 为每个模型生成视频
    success_count = 0
    for model_path in model_files:
        if generate_video_for_model(model_path, env_name, config.MAX_STEPS):
            success_count += 1
    
    print()
    print("=" * 70)
    print(f"完成！成功生成 {success_count}/{len(model_files)} 个视频")
    print("=" * 70)
    print(f"\n视频保存位置: videos/ 目录")
    print("\n你现在可以对比不同代数的训练效果了！")
    
    # 列出生成的视频
    video_files = sorted(glob.glob("videos/**/*.mp4", recursive=True))
    if video_files:
        print(f"\n生成的视频文件 ({len(video_files)}个):")
        for vf in video_files:
            size_mb = os.path.getsize(vf) / (1024*1024)
            print(f"  - {vf} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

