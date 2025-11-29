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
    
    # 时间预算：约为总时间的40%
    total_time = sum(c.time_cost for c in courses)
    time_budget = int(total_time * 0.4)
    
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
        """初始化种群（随机二进制串）"""
        # 随机生成二进制个体
        population = np.random.randint(0, 2, size=(self.population_size, self.n_courses))
        
        # 确保至少有一些个体是可行的（不超时）
        for i in range(min(10, self.population_size)):
            # 贪心初始化：按性价比排序
            value_per_time = [c.value / c.time_cost for c in self.courses]
            sorted_indices = np.argsort(value_per_time)[::-1]
            
            individual = np.zeros(self.n_courses, dtype=int)
            current_time = 0
            for idx in sorted_indices:
                if current_time + self.courses[idx].time_cost <= self.time_budget:
                    individual[idx] = 1
                    current_time += self.courses[idx].time_cost
            
            population[i] = individual
        
        return population
    
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
        
        # 如果超时，施加惩罚
        if total_time > self.time_budget:
            overtime = total_time - self.time_budget
            penalty = overtime * 10  # 每超时1小时，惩罚10分
            fitness = total_value - penalty
        else:
            fitness = total_value
        
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
        """单点交叉"""
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.n_courses)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """位翻转变异"""
        for i in range(self.n_courses):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # 0->1 或 1->0
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
    """可视化训练结果"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 图1：适应度曲线
    ax1.plot(range(1, generations + 1), ga.best_fitness_history, 
             'b-', linewidth=2, label='最佳适应度')
    ax1.plot(range(1, generations + 1), ga.avg_fitness_history, 
             'r--', linewidth=1, alpha=0.7, label='平均适应度')
    ax1.set_xlabel('代数', fontsize=12)
    ax1.set_ylabel('适应度（总收获）', fontsize=12)
    ax1.set_title('遗传算法训练曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2：最佳方案详情
    solution = ga.get_best_solution()
    
    # 绘制条形图
    course_names = [c.name for c in solution['courses'][:10]]  # 只显示前10个
    course_values = [c.value for c in solution['courses'][:10]]
    course_times = [c.time_cost for c in solution['courses'][:10]]
    
    x = np.arange(len(course_names))
    width = 0.35
    
    ax2.bar(x - width/2, course_values, width, label='收获（分）', alpha=0.8)
    ax2.bar(x + width/2, course_times, width, label='时间（小时）', alpha=0.8)
    ax2.set_xlabel('课程', fontsize=12)
    ax2.set_ylabel('数值', fontsize=12)
    ax2.set_title(f'选中课程详情（前10门，共{len(solution["courses"])}门）', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(course_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
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
    N_COURSES = 50          # 课程数量
    POPULATION_SIZE = 100   # 种群大小
    GENERATIONS = 200       # 进化代数
    MUTATION_RATE = 0.01    # 变异率
    CROSSOVER_RATE = 0.8    # 交叉率
    ELITE_RATIO = 0.1       # 精英比例
    
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
    print(f"时间预算: {time_budget} 小时 (约40%)")
    
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
        
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"[Gen {generation+1:3d}/{GENERATIONS}] "
                  f"最佳: {ga.best_fitness:.0f}分 | "
                  f"平均: {np.mean(ga.fitness_scores):.0f}分")
    
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

