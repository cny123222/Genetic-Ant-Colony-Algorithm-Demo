"""
蚁群算法求解旅行商问题(TSP)
适合生命科学导论课程Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AntColony:
    """蚁群算法"""
    
    def __init__(self, cities, n_ants=20, n_iterations=100, 
                 alpha=1.0, beta=2.0, evaporation=0.5, Q=100):
        """
        Parameters:
        - cities: 城市坐标列表 [(x1,y1), (x2,y2), ...]
        - n_ants: 蚂蚁数量
        - n_iterations: 迭代次数
        - alpha: 信息素重要程度
        - beta: 启发式因子重要程度
        - evaporation: 信息素挥发率
        - Q: 信息素强度
        """
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        
        # 计算距离矩阵
        self.distances = self._calculate_distances()
        
        # 初始化信息素矩阵
        self.pheromones = np.ones((self.n_cities, self.n_cities))
        
        # 记录历史
        self.best_path = None
        self.best_distance = float('inf')
        self.history = []
        
    def _calculate_distances(self):
        """计算城市间距离矩阵"""
        n = self.n_cities
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                distances[i][j] = distances[j][i] = dist
        return distances
    
    def _select_next_city(self, current, unvisited):
        """选择下一个城市（轮盘赌选择）"""
        pheromone = np.array([self.pheromones[current][city] for city in unvisited])
        heuristic = np.array([1.0 / self.distances[current][city] if self.distances[current][city] > 0 else 0 
                             for city in unvisited])
        
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities = probabilities / probabilities.sum()
        
        next_city = np.random.choice(unvisited, p=probabilities)
        return next_city
    
    def _construct_solution(self):
        """一只蚂蚁构造一个解"""
        path = [random.randint(0, self.n_cities-1)]
        unvisited = list(range(self.n_cities))
        unvisited.remove(path[0])
        
        while unvisited:
            next_city = self._select_next_city(path[-1], unvisited)
            path.append(next_city)
            unvisited.remove(next_city)
        
        return path
    
    def _calculate_path_distance(self, path):
        """计算路径总距离"""
        distance = 0
        for i in range(len(path)):
            distance += self.distances[path[i]][path[(i+1) % len(path)]]
        return distance
    
    def _update_pheromones(self, all_paths):
        """更新信息素"""
        # 挥发
        self.pheromones *= (1 - self.evaporation)
        
        # 添加新信息素
        for path in all_paths:
            distance = self._calculate_path_distance(path)
            for i in range(len(path)):
                city_a = path[i]
                city_b = path[(i+1) % len(path)]
                self.pheromones[city_a][city_b] += self.Q / distance
                self.pheromones[city_b][city_a] += self.Q / distance
    
    def run(self):
        """运行蚁群算法"""
        for iteration in range(self.n_iterations):
            # 所有蚂蚁构造解
            all_paths = []
            for ant in range(self.n_ants):
                path = self._construct_solution()
                all_paths.append(path)
                
                # 更新最佳解
                distance = self._calculate_path_distance(path)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # 更新信息素
            self._update_pheromones(all_paths)
            
            # 记录历史
            self.history.append({
                'iteration': iteration,
                'best_distance': self.best_distance,
                'best_path': self.best_path.copy(),
                'pheromones': self.pheromones.copy()
            })
            
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration+1}/{self.n_iterations}, "
                      f"最佳距离: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance


def visualize_evolution(cities, history, save_path='tsp_evolution.gif'):
    """
    可视化TSP求解过程
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 城市坐标
    cities = np.array(cities)
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        data = history[frame]
        path = data['best_path']
        pheromones = data['pheromones']
        
        # 左图：路径演化
        ax1.set_title(f'迭代 {frame+1}: 最佳路径\n距离: {data["best_distance"]:.2f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlim(-0.1, 1.1)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_aspect('equal')
        
        # 绘制城市
        ax1.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=5)
        for i, (x, y) in enumerate(cities):
            ax1.text(x, y, str(i), ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=10)
        
        # 绘制路径
        for i in range(len(path)):
            city_a = path[i]
            city_b = path[(i+1) % len(path)]
            ax1.plot([cities[city_a][0], cities[city_b][0]],
                    [cities[city_a][1], cities[city_b][1]],
                    'b-', linewidth=2, alpha=0.6)
        
        # 右图：信息素热力图
        ax2.set_title('信息素分布热力图', fontsize=14, fontweight='bold')
        im = ax2.imshow(pheromones, cmap='YlOrRd', interpolation='nearest')
        ax2.set_xlabel('城市')
        ax2.set_ylabel('城市')
        
        if frame == 0:
            plt.colorbar(im, ax=ax2, label='信息素浓度')
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(history), 
                                 interval=200, repeat=True)
    
    # 保存
    ani.save(save_path, writer='pillow', fps=5)
    print(f"动画已保存到: {save_path}")
    
    plt.close()


def generate_demo():
    """生成Demo"""
    print("=" * 60)
    print("蚁群算法求解旅行商问题 - 生命科学导论Demo")
    print("=" * 60)
    
    # 生成20个随机城市
    np.random.seed(42)
    n_cities = 20
    cities = np.random.rand(n_cities, 2)
    
    print(f"\n生成{n_cities}个随机城市...")
    print(f"蚂蚁数量: 20")
    print(f"迭代次数: 100\n")
    
    # 运行蚁群算法
    aco = AntColony(cities, n_ants=20, n_iterations=100)
    best_path, best_distance = aco.run()
    
    print(f"\n{'='*60}")
    print(f"求解完成！")
    print(f"最佳路径: {best_path}")
    print(f"最短距离: {best_distance:.2f}")
    print(f"{'='*60}\n")
    
    # 生成可视化（每10次迭代记录一次）
    print("生成可视化动画...")
    history_sample = [aco.history[i] for i in range(0, len(aco.history), 5)]
    visualize_evolution(cities, history_sample, 'ant_colony_tsp_demo.gif')
    
    # 静态对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 初始随机路径
    random_path = list(range(n_cities))
    random.shuffle(random_path)
    random_distance = aco._calculate_path_distance(random_path)
    
    # 左图：随机路径
    ax1.set_title(f'初始随机路径\n距离: {random_distance:.2f}', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=5)
    for i in range(len(random_path)):
        city_a = random_path[i]
        city_b = random_path[(i+1) % len(random_path)]
        ax1.plot([cities[city_a][0], cities[city_b][0]],
                [cities[city_a][1], cities[city_b][1]],
                'gray', linewidth=2, alpha=0.5, linestyle='--')
    
    # 右图：优化后路径
    ax2.set_title(f'蚁群优化后路径\n距离: {best_distance:.2f}', 
                 fontsize=14, fontweight='bold', color='green')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect('equal')
    ax2.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=5)
    for i in range(len(best_path)):
        city_a = best_path[i]
        city_b = best_path[(i+1) % len(best_path)]
        ax2.plot([cities[city_a][0], cities[city_b][0]],
                [cities[city_a][1], cities[city_b][1]],
                'green', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ant_colony_comparison.png', dpi=150, bbox_inches='tight')
    print("对比图已保存: ant_colony_comparison.png")
    
    print("\n✅ Demo生成完成！")
    print(f"改进: {(random_distance - best_distance) / random_distance * 100:.1f}%")


if __name__ == '__main__':
    generate_demo()

