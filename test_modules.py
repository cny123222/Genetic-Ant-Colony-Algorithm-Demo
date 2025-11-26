"""
测试脚本：验证所有模块是否正常工作
运行此脚本来检查环境配置是否正确
"""

import sys
import numpy as np


def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # 测试NumPy
    tests_total += 1
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ NumPy: {e}")
    
    # 测试Matplotlib
    tests_total += 1
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
    
    # 测试Gymnasium
    tests_total += 1
    try:
        import gymnasium
        print(f"✓ Gymnasium {gymnasium.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Gymnasium: {e}")
    
    # 测试MuJoCo（可选）
    tests_total += 1
    try:
        import mujoco
        print(f"✓ MuJoCo {mujoco.__version__}")
        tests_passed += 1
    except ImportError:
        print(f"⚠  MuJoCo: 未安装（可选，但推荐安装以使用Humanoid环境）")
    
    print()
    return tests_passed, tests_total


def test_custom_modules():
    """测试自定义模块"""
    print("=" * 60)
    print("测试自定义模块")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    # 测试神经网络模块
    try:
        from neural_network import NeuralNetwork, create_random_params
        nn = NeuralNetwork(10, [32], 4)
        params = create_random_params(nn.get_param_count())
        nn.set_params(params)
        output = nn.predict(np.random.randn(10))
        assert output.shape == (4,)
        print("✓ neural_network.py 正常")
        tests_passed += 1
    except Exception as e:
        print(f"✗ neural_network.py: {e}")
    
    # 测试遗传算法模块
    try:
        from genetic_algorithm import GeneticAlgorithm
        ga = GeneticAlgorithm(population_size=10, param_count=50)
        fitness = lambda x: np.sum(x**2)
        ga.evaluate_population(fitness)
        stats = ga.get_statistics()
        ga.evolve()
        print("✓ genetic_algorithm.py 正常")
        tests_passed += 1
    except Exception as e:
        print(f"✗ genetic_algorithm.py: {e}")
    
    # 测试配置模块
    try:
        import config
        assert hasattr(config, 'POPULATION_SIZE')
        assert hasattr(config, 'GENERATIONS')
        print("✓ config.py 正常")
        tests_passed += 1
    except Exception as e:
        print(f"✗ config.py: {e}")
    
    print()
    return tests_passed, tests_total


def test_environments():
    """测试Gymnasium环境"""
    print("=" * 60)
    print("测试Gymnasium环境")
    print("=" * 60)
    
    try:
        import gymnasium as gym
    except ImportError:
        print("✗ Gymnasium未安装，跳过环境测试")
        return 0, 0
    
    environments = [
        ('CartPole-v1', '简单', '⭐'),
        ('BipedalWalker-v3', '中等', '⭐⭐⭐'),
        ('Humanoid-v4', '困难', '⭐⭐⭐⭐⭐')
    ]
    
    available = 0
    
    for env_name, difficulty, stars in environments:
        try:
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            env.close()
            print(f"✓ {env_name:20s} | 难度: {difficulty:4s} {stars:6s} | "
                  f"观察: {obs_dim:3d}维, 动作: {act_dim:2d}维")
            available += 1
        except Exception as e:
            print(f"✗ {env_name:20s} | {str(e)[:40]}")
    
    print()
    return available, len(environments)


def test_integration():
    """集成测试：运行一个完整的小规模训练"""
    print("=" * 60)
    print("集成测试：小规模训练")
    print("=" * 60)
    
    try:
        import gymnasium as gym
        from neural_network import NeuralNetwork
        from genetic_algorithm import GeneticAlgorithm
        
        # 使用CartPole进行快速测试
        print("使用CartPole-v1进行5代快速训练...")
        env = gym.make('CartPole-v1')
        
        # 创建网络
        obs_dim = 4
        act_dim = 1
        network = NeuralNetwork(obs_dim, [16], act_dim)
        
        # 创建GA
        ga = GeneticAlgorithm(
            population_size=10,
            param_count=network.get_param_count(),
            mutation_rate=0.2,
            crossover_rate=0.7
        )
        
        # 评估函数
        def evaluate(params):
            network.set_params(params)
            observation, _ = env.reset()
            total_reward = 0
            for _ in range(200):
                action = 1 if network.predict(observation)[0] > 0 else 0
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            return total_reward
        
        # 训练5代
        print()
        for gen in range(5):
            ga.evaluate_population(evaluate)
            stats = ga.get_statistics()
            print(f"  第 {gen+1} 代 | 最佳: {stats['best']:5.1f}, 平均: {stats['mean']:5.1f}")
            ga.evolve()
        
        env.close()
        
        print()
        print("✓ 集成测试通过")
        print()
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "模块测试与环境检查" + " " * 23 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # 测试导入
    import_passed, import_total = test_imports()
    
    # 测试自定义模块
    module_passed, module_total = test_custom_modules()
    
    # 测试环境
    env_available, env_total = test_environments()
    
    # 集成测试
    integration_ok = test_integration()
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"依赖库: {import_passed}/{import_total} 通过")
    print(f"自定义模块: {module_passed}/{module_total} 通过")
    print(f"可用环境: {env_available}/{env_total}")
    print(f"集成测试: {'通过' if integration_ok else '失败'}")
    print()
    
    # 建议
    if import_passed < import_total:
        print("⚠️  建议：安装缺失的依赖库")
        print("   运行: pip install -r requirements.txt")
        print()
    
    if env_available == 0:
        print("⚠️  警告：没有可用的环境")
        print("   请确保 Gymnasium 正确安装")
        print()
    elif env_available < env_total:
        print("ℹ️  提示：部分环境不可用")
        if env_available >= 1:
            print("   但至少有一个环境可用，可以开始训练")
        print()
    
    if integration_ok:
        print("🎉 所有测试通过！环境配置正确。")
        print()
        print("下一步：")
        print("  1. 快速演示: python demo_cartpole.py")
        print("  2. 开始训练: python train_simple.py")
        print("  3. 查看配置: python config.py")
        print()
    else:
        print("❌ 测试未完全通过，请检查错误信息并修复")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n\n测试过程发生错误: {e}")
        import traceback
        traceback.print_exc()

