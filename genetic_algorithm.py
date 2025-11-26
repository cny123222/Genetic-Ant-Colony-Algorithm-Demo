"""
遗传算法模块
实现基本的遗传算法操作：选择、交叉、变异
"""

import numpy as np
from typing import List, Tuple, Callable


class GeneticAlgorithm:
    """遗传算法类"""
    
    def __init__(
        self,
        population_size: int,
        param_count: int,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.3,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        tournament_size: int = 3
    ):
        """
        初始化遗传算法
        
        Args:
            population_size: 种群大小
            param_count: 每个个体的参数数量
            mutation_rate: 变异率（每个参数变异的概率）
            mutation_scale: 变异幅度（高斯噪声的标准差）
            crossover_rate: 交叉率（发生交叉的概率）
            elite_ratio: 精英比例（直接保留到下一代的最优个体比例）
            tournament_size: 锦标赛选择的参赛个体数量
        """
        self.population_size = population_size
        self.param_count = param_count
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        
        # 计算精英数量
        self.elite_count = max(1, int(population_size * elite_ratio))
        
        # 初始化种群
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(population_size)
    
    def _initialize_population(self) -> np.ndarray:
        """
        初始化种群（随机生成）
        
        Returns:
            种群矩阵 (population_size, param_count)
        """
        # 使用 Xavier 初始化
        scale = np.sqrt(2.0 / self.param_count)
        population = np.random.randn(self.population_size, self.param_count) * scale
        return population
    
    def evaluate_population(self, fitness_func: Callable) -> np.ndarray:
        """
        评估种群中所有个体的适应度
        
        Args:
            fitness_func: 适应度评估函数，接受参数向量，返回适应度分数
            
        Returns:
            适应度分数数组
        """
        self.fitness_scores = np.array([
            fitness_func(individual) for individual in self.population
        ])
        return self.fitness_scores
    
    def get_best_individual(self) -> Tuple[np.ndarray, float]:
        """
        获取当前种群中的最佳个体
        
        Returns:
            (最佳个体参数, 最佳适应度分数)
        """
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy(), self.fitness_scores[best_idx]
    
    def get_statistics(self) -> dict:
        """
        获取当前种群的统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'best': np.max(self.fitness_scores),
            'mean': np.mean(self.fitness_scores),
            'std': np.std(self.fitness_scores),
            'worst': np.min(self.fitness_scores)
        }
    
    def tournament_selection(self) -> np.ndarray:
        """
        锦标赛选择
        
        Returns:
            选中的个体
        """
        # 随机选择参赛个体
        tournament_indices = np.random.choice(
            self.population_size,
            size=self.tournament_size,
            replace=False
        )
        # 选择其中适应度最高的
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        交叉操作（均匀交叉）
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            (子代1, 子代2)
        """
        if np.random.random() > self.crossover_rate:
            # 不进行交叉，直接返回父代副本
            return parent1.copy(), parent2.copy()
        
        # 均匀交叉：随机选择每个参数来自哪个父代
        mask = np.random.random(self.param_count) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作（高斯噪声）
        
        Args:
            individual: 个体参数
            
        Returns:
            变异后的个体
        """
        # 生成变异掩码
        mutation_mask = np.random.random(self.param_count) < self.mutation_rate
        
        # 添加高斯噪声
        noise = np.random.randn(self.param_count) * self.mutation_scale
        mutated = individual + mutation_mask * noise
        
        return mutated
    
    def evolve(self) -> None:
        """
        进化到下一代
        
        执行选择、交叉、变异操作，生成新一代种群
        """
        # 按适应度排序
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # 保留精英
        new_population = [self.population[idx].copy() for idx in sorted_indices[:self.elite_count]]
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # 添加到新种群
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # 更新种群
        self.population = np.array(new_population[:self.population_size])
    
    def get_population(self) -> np.ndarray:
        """返回当前种群"""
        return self.population.copy()
    
    def set_population(self, population: np.ndarray) -> None:
        """设置种群"""
        assert population.shape == (self.population_size, self.param_count)
        self.population = population.copy()


class AdaptiveGeneticAlgorithm(GeneticAlgorithm):
    """自适应遗传算法（可选的高级版本）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_mutation_rate = self.mutation_rate
        self.initial_mutation_scale = self.mutation_scale
        self.stagnation_counter = 0
        self.last_best_fitness = -np.inf
    
    def adapt_parameters(self):
        """根据进化情况自适应调整参数"""
        current_best = np.max(self.fitness_scores)
        
        # 检测停滞
        if abs(current_best - self.last_best_fitness) < 0.01:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # 如果停滞，增加探索性
        if self.stagnation_counter > 5:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
            self.mutation_scale = min(0.5, self.mutation_scale * 1.1)
        else:
            # 逐渐降低变异率，增加收敛性
            self.mutation_rate = max(0.05, self.mutation_rate * 0.99)
            self.mutation_scale = max(0.1, self.mutation_scale * 0.99)
        
        self.last_best_fitness = current_best
    
    def evolve(self):
        """进化并自适应调整参数"""
        super().evolve()
        self.adapt_parameters()


if __name__ == "__main__":
    # 测试遗传算法
    print("测试遗传算法模块...")
    
    # 定义一个简单的适应度函数（最大化参数和）
    def simple_fitness(params):
        return np.sum(params ** 2)
    
    # 创建遗传算法实例
    ga = GeneticAlgorithm(
        population_size=20,
        param_count=10,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    print(f"初始种群形状: {ga.population.shape}")
    
    # 运行几代
    for gen in range(10):
        ga.evaluate_population(simple_fitness)
        stats = ga.get_statistics()
        print(f"第 {gen+1} 代 - 最佳: {stats['best']:.2f}, 平均: {stats['mean']:.2f}")
        ga.evolve()
    
    # 测试自适应版本
    print("\n测试自适应遗传算法...")
    aga = AdaptiveGeneticAlgorithm(
        population_size=20,
        param_count=10
    )
    
    for gen in range(10):
        aga.evaluate_population(simple_fitness)
        stats = aga.get_statistics()
        print(f"第 {gen+1} 代 - 最佳: {stats['best']:.2f}, 变异率: {aga.mutation_rate:.3f}")
        aga.evolve()
    
    print("测试完成！")

