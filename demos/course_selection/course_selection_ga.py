"""
遗传算法解决选课问题（0-1背包问题）

问题描述：
- 有N门课程，每门课有预期投入时间和预期收获
- 时间预算有限
- 目标：在时间约束下最大化总收获

编码方式：
- 二进制编码，每个基因表示是否选择该课程（0或1）
- 染色体长度 = 课程数量N
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json


class Course:
    """课程类"""
    def __init__(self, name, time_cost, value):
        self.name = name           # 课程名称
        self.time_cost = time_cost # 时间成本（小时）
        self.value = value         # 预期收获（分数）
    
    def __repr__(self):
        return f"{self.name}(时间:{self.time_cost}h, 收获:{self.value}分)"


def generate_courses(n=50):
    """
    生成N门课程
    
    Args:
        n: 课程数量
    
    Returns:
        courses: 课程列表
        time_budget: 时间预算
    """
    np.random.seed(42)  # 固定随机种子，保证可复现
    
    courses = []
    for i in range(n):
        # 时间成本：10-50小时
        time_cost = np.random.randint(10, 51)
        
        # 收获：30-100分（大致与时间成本正相关，但有随机性）
        base_value = time_cost * (0.8 + np.random.rand() * 0.8)  # 0.8-1.6倍
        value = int(base_value)
        
        course = Course(f"课程{i+1:02d}", time_cost, value)
        courses.append(course)
    
    # 时间预算：约为总时间的15%（非常严格的约束）
    total_time = sum(c.time_cost for c in courses)
    time_budget = int(total_time * 0.15)
    
    return courses, time_budget


class CourseSelectionGA:
    """选课问题的遗传算法"""
    
    def __init__(self, courses, time_budget, population_size=100, 
                 mutation_rate=0.01, crossover_rate=0.8, elite_ratio=0.1):
        """
        初始化遗传算法
        
        Args:
            courses: 课程列表
            time_budget: 时间预算
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_ratio: 精英比例
        """
        self.courses = courses
        self.time_budget = time_budget
        self.n_courses = len(courses)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = int(population_size * elite_ratio)
        
        # 初始化种群（二进制编码）
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(population_size)
        
        # 统计信息
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
    
    def _initialize_population(self):
        """初始化种群（随机策略）"""
        population = []
        
        # 完全随机初始化
        for i in range(self.population_size):
            # 随机选择，期望选中概率约为 time_budget / total_time
            total_time = sum(c.time_cost for c in self.courses)
            select_prob = self.time_budget / total_time * 1.2  # 稍微多选一些，后面修复
            
            individual = (np.random.rand(self.n_courses) < select_prob).astype(int)
            
            # 轻度修复：只移除明显超时的情况
            current_time = sum(self.courses[j].time_cost for j in range(self.n_courses) if individual[j] == 1)
            
            # 如果严重超时（>1.5倍），随机移除一些
            while current_time > self.time_budget * 1.5:
                selected = [j for j in range(self.n_courses) if individual[j] == 1]
                if not selected:
                    break
                remove_idx = np.random.choice(selected)
                individual[remove_idx] = 0
                current_time -= self.courses[remove_idx].time_cost
            
            population.append(individual)
        
        return np.array(population)
    
    def _calculate_fitness(self, individual):
        """
        计算个体适应度
        
        适应度 = 总收获
        如果超时，施加惩罚：适应度 = 总收获 - 超时惩罚
        
        Args:
            individual: 二进制编码的个体
        
        Returns:
            fitness: 适应度值
        """
        total_time = 0
        total_value = 0
        
        for i, selected in enumerate(individual):
            if selected == 1:
                total_time += self.courses[i].time_cost
                total_value += self.courses[i].value
        
        # 如果超时，施加强惩罚
        if total_time > self.time_budget:
            overtime = total_time - self.time_budget
            penalty = overtime * 20  # 每超时1小时，惩罚20分（加强）
            fitness = total_value - penalty
        else:
            # 可行解：奖励时间利用率
            utilization_bonus = (total_time / self.time_budget) * 5
            fitness = total_value + utilization_bonus
        
        return fitness
    
    def evaluate_population(self):
        """评估整个种群"""
        for i, individual in enumerate(self.population):
            self.fitness_scores[i] = self._calculate_fitness(individual)
        
        # 更新最佳个体
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_individual = self.population[best_idx].copy()
    
    def selection(self):
        """锦标赛选择"""
        tournament_size = 5
        selected_idx = np.random.choice(self.population_size, tournament_size, replace=False)
        tournament_fitness = self.fitness_scores[selected_idx]
        winner_idx = selected_idx[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """两点交叉（更好的基因混合）"""
        if np.random.rand() < self.crossover_rate:
            # 两点交叉：选择两个交叉点
            point1 = np.random.randint(1, self.n_courses - 1)
            point2 = np.random.randint(point1 + 1, self.n_courses)
            
            # 中间段交换
            child1 = parent1.copy()
            child2 = parent2.copy()
            child1[point1:point2] = parent2[point1:point2]
            child2[point1:point2] = parent1[point1:point2]
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """位翻转变异（轻度修复）"""
        # 位翻转变异
        for i in range(self.n_courses):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # 0->1 或 1->0
        
        # 轻度修复：只修复严重超时的情况（>2倍预算）
        total_time = sum(self.courses[i].time_cost for i in range(self.n_courses) if individual[i] == 1)
        attempts = 0
        while total_time > self.time_budget * 2 and attempts < 20:
            selected_indices = [i for i in range(self.n_courses) if individual[i] == 1]
            if not selected_indices:
                break
            # 随机移除一个选中的课程
            remove_idx = np.random.choice(selected_indices)
            individual[remove_idx] = 0
            total_time -= self.courses[remove_idx].time_cost
            attempts += 1
        
        return individual
    
    def evolve(self):
        """进化一代"""
        # 评估当前种群
        self.evaluate_population()
        
        # 记录统计信息
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(np.mean(self.fitness_scores))
        
        # 精英保留
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_count:]
        elites = self.population[elite_indices].copy()
        
        # 生成新一代
        new_population = []
        
        # 保留精英
        for elite in elites:
            new_population.append(elite)
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择
            parent1 = self.selection()
            parent2 = self.selection()
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = np.array(new_population)
    
    def get_best_solution(self):
        """获取最佳解决方案"""
        selected_courses = []
        total_time = 0
        total_value = 0
        
        for i, selected in enumerate(self.best_individual):
            if selected == 1:
                course = self.courses[i]
                selected_courses.append(course)
                total_time += course.time_cost
                total_value += course.value
        
        return {
            'courses': selected_courses,
            'total_time': total_time,
            'total_value': total_value,
            'time_budget': self.time_budget,
            'utilization': total_time / self.time_budget * 100
        }


def visualize_results(ga, generations):
    """可视化训练结果（仅显示最佳适应度曲线）"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建单个图表（增大高度）
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制最佳适应度曲线
    ax.plot(range(1, generations + 1), ga.best_fitness_history, 
            'b-', linewidth=2.5, label='最佳适应度')
    
    ax.set_xlabel('代数', fontsize=16)
    ax.set_ylabel('适应度（总收获）', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 增大刻度字体
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"course_selection_result_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {filename}")
    
    plt.show()


def print_solution(solution):
    """打印最佳解决方案"""
    print("\n" + "=" * 80)
    print("最佳选课方案")
    print("=" * 80)
    print(f"时间预算: {solution['time_budget']} 小时")
    print(f"实际使用: {solution['total_time']} 小时 ({solution['utilization']:.1f}%)")
    print(f"总收获: {solution['total_value']} 分")
    print(f"选中课程数: {len(solution['courses'])} 门")
    print("\n选中的课程列表：")
    print("-" * 80)
    
    # 按收获排序
    sorted_courses = sorted(solution['courses'], key=lambda c: c.value, reverse=True)
    
    for i, course in enumerate(sorted_courses, 1):
        ratio = course.value / course.time_cost
        print(f"{i:2d}. {course.name:8s} | 时间: {course.time_cost:2d}h | "
              f"收获: {course.value:3d}分 | 性价比: {ratio:.2f}")
    
    print("=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("遗传算法解决选课问题（0-1背包问题）")
    print("=" * 80)
    
    # 参数设置
    N_COURSES = 200         # 课程数量
    POPULATION_SIZE = 200   # 种群大小
    GENERATIONS = 1000      # 进化代数
    MUTATION_RATE = 0.02    # 变异率
    CROSSOVER_RATE = 0.85   # 交叉率
    ELITE_RATIO = 0.05      # 精英比例
    
    print(f"\n配置参数：")
    print(f"  课程数量: {N_COURSES}")
    print(f"  种群大小: {POPULATION_SIZE}")
    print(f"  进化代数: {GENERATIONS}")
    print(f"  变异率: {MUTATION_RATE}")
    print(f"  交叉率: {CROSSOVER_RATE}")
    print(f"  精英比例: {ELITE_RATIO}")
    
    # 生成课程
    print("\n正在生成课程数据...")
    courses, time_budget = generate_courses(N_COURSES)
    
    print(f"\n✅ 生成了 {len(courses)} 门课程")
    print(f"总时间: {sum(c.time_cost for c in courses)} 小时")
    print(f"总收获: {sum(c.value for c in courses)} 分")
    print(f"时间预算: {time_budget} 小时 (约15%，非常严格的约束⚠️)")
    
    # 显示部分课程
    print("\n课程样例（前10门）：")
    for course in courses[:10]:
        ratio = course.value / course.time_cost
        print(f"  {course.name}: {course.time_cost}h → {course.value}分 (性价比: {ratio:.2f})")
    
    # 创建遗传算法
    print(f"\n{'='*80}")
    print("开始遗传算法优化...")
    print(f"{'='*80}\n")
    
    ga = CourseSelectionGA(
        courses=courses,
        time_budget=time_budget,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elite_ratio=ELITE_RATIO
    )
    
    # 进化
    for generation in range(GENERATIONS):
        ga.evolve()
        
        # 每50代打印一次，初始也打印
        if (generation + 1) % 50 == 0 or generation == 0:
            print(f"[Gen {generation+1:3d}/{GENERATIONS}] 最佳适应度: {ga.best_fitness:.1f}分")
    
    print(f"\n{'='*80}")
    print("优化完成！")
    print(f"{'='*80}")
    
    # 获取并打印最佳方案
    solution = ga.get_best_solution()
    print_solution(solution)
    
    # 可视化结果
    print("\n正在生成可视化图表...")
    visualize_results(ga, GENERATIONS)
    
    # 保存结果到JSON
    result_data = {
        'parameters': {
            'n_courses': N_COURSES,
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'time_budget': time_budget
        },
        'solution': {
            'total_value': solution['total_value'],
            'total_time': solution['total_time'],
            'n_courses_selected': len(solution['courses']),
            'utilization': solution['utilization']
        },
        'best_fitness_history': ga.best_fitness_history
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"course_selection_result_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 结果已保存: {json_filename}")
    
    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()

