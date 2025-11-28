"""
快速测试不同随机种子，找到能达到好成绩的种子
只训练100代来快速筛选
"""

import subprocess
import os
import time

# 测试的种子列表
SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 7777, 9999]

print("=" * 60)
print("快速种子筛选测试")
print("=" * 60)
print(f"将测试{len(SEEDS)}个种子，每个训练100代")
print("目标：找到前100代fitness最高的种子\n")

results = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"测试种子: {seed}")
    print(f"{'='*60}")
    
    # 修改config.py中的种子和代数
    with open('config.py', 'r') as f:
        config_content = f.read()
    
    # 临时修改为100代快速测试
    config_content = config_content.replace('GENERATIONS = 600', 'GENERATIONS = 100')
    config_content = config_content.replace(f'RANDOM_SEED = 42', f'RANDOM_SEED = {seed}')
    
    with open('config_temp.py', 'w') as f:
        f.write(config_content)
    
    # 备份原config
    os.rename('config.py', 'config_backup.py')
    os.rename('config_temp.py', 'config.py')
    
    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'ga-humanoid', 'python', 'train_with_video.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        # 提取最佳fitness
        output = result.stdout
        if 'NEW RECORD' in output:
            records = [line for line in output.split('\n') if 'NEW RECORD' in line]
            if records:
                last_record = records[-1]
                # 提取fitness值
                if 'Best:' in last_record:
                    fitness_str = last_record.split('Best:')[1].split(',')[0].strip()
                    best_fitness = float(fitness_str)
                    results.append((seed, best_fitness))
                    print(f"✅ 种子{seed}: 最佳fitness = {best_fitness:.2f}")
                else:
                    results.append((seed, -999))
                    print(f"⚠️ 种子{seed}: 解析失败")
        else:
            results.append((seed, -999))
            print(f"❌ 种子{seed}: 训练失败")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 种子{seed}: 超时")
        results.append((seed, -999))
    except Exception as e:
        print(f"❌ 种子{seed}: 错误 - {e}")
        results.append((seed, -999))
    finally:
        # 恢复原config
        os.rename('config.py', 'config_temp.py')
        os.rename('config_backup.py', 'config.py')
    
    elapsed = time.time() - start_time
    print(f"用时: {elapsed/60:.1f}分钟")

# 恢复原配置
print(f"\n\n{'='*60}")
print("测试完成！结果汇总：")
print(f"{'='*60}\n")

results.sort(key=lambda x: x[1], reverse=True)
for seed, fitness in results:
    if fitness > -999:
        print(f"种子 {seed:5d}: {fitness:7.2f}")
    else:
        print(f"种子 {seed:5d}: 失败")

if results and results[0][1] > -999:
    best_seed = results[0][0]
    best_fitness = results[0][1]
    print(f"\n🏆 最佳种子: {best_seed}, fitness: {best_fitness:.2f}")
    print(f"\n建议在config.py中设置: RANDOM_SEED = {best_seed}")

