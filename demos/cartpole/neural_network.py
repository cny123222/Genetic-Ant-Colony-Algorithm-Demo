"""
神经网络控制器模块
实现简单的前馈神经网络，用于控制机器人动作
"""

import numpy as np


class NeuralNetwork:
    """简单的前馈神经网络控制器"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        初始化神经网络
        
        Args:
            input_size: 输入层大小（观察空间维度）
            hidden_sizes: 隐藏层大小列表，如 [64, 32]
            output_size: 输出层大小（动作空间维度）
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # 计算网络结构
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # 计算总参数数量
        self.param_count = 0
        for i in range(len(self.layer_sizes) - 1):
            # 权重 + 偏置
            self.param_count += self.layer_sizes[i] * self.layer_sizes[i + 1]
            self.param_count += self.layer_sizes[i + 1]
        
        # 初始化参数为零（将由遗传算法设置）
        self.params = np.zeros(self.param_count)
    
    def get_param_count(self):
        """返回网络参数总数"""
        return self.param_count
    
    def set_params(self, params):
        """
        设置网络参数（从扁平化向量）
        
        Args:
            params: 一维参数向量
        """
        assert len(params) == self.param_count, \
            f"参数数量不匹配: 期望 {self.param_count}, 得到 {len(params)}"
        self.params = params.copy()
    
    def get_params(self):
        """返回网络参数的副本"""
        return self.params.copy()
    
    def _unflatten_params(self):
        """将扁平化的参数转换为权重和偏置矩阵"""
        weights = []
        biases = []
        idx = 0
        
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            
            # 提取权重矩阵
            weight_size = in_size * out_size
            weight = self.params[idx:idx + weight_size].reshape(in_size, out_size)
            weights.append(weight)
            idx += weight_size
            
            # 提取偏置向量
            bias = self.params[idx:idx + out_size]
            biases.append(bias)
            idx += out_size
        
        return weights, biases
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入向量（观察）
            
        Returns:
            输出向量（动作）
        """
        weights, biases = self._unflatten_params()
        
        # 前向传播
        activation = x
        for i in range(len(weights)):
            z = np.dot(activation, weights[i]) + biases[i]
            # 使用 tanh 作为激活函数
            activation = np.tanh(z)
        
        return activation
    
    def predict(self, observation):
        """
        根据观察预测动作
        
        Args:
            observation: 环境观察向量
            
        Returns:
            动作向量
        """
        return self.forward(observation)


def create_random_params(param_count, scale=1.0):
    """
    创建随机初始化的参数向量
    
    Args:
        param_count: 参数数量
        scale: 初始化规模
        
    Returns:
        随机参数向量
    """
    return np.random.randn(param_count) * scale


def save_network(network, filename):
    """
    保存神经网络参数到文件
    
    Args:
        network: NeuralNetwork 实例
        filename: 文件路径
    """
    data = {
        'input_size': network.input_size,
        'hidden_sizes': network.hidden_sizes,
        'output_size': network.output_size,
        'params': network.params
    }
    np.save(filename, data)
    print(f"模型已保存到 {filename}")


def load_network(filename):
    """
    从文件加载神经网络
    
    Args:
        filename: 文件路径
        
    Returns:
        NeuralNetwork 实例
    """
    data = np.load(filename, allow_pickle=True).item()
    network = NeuralNetwork(
        data['input_size'],
        data['hidden_sizes'],
        data['output_size']
    )
    network.set_params(data['params'])
    print(f"模型已从 {filename} 加载")
    return network


if __name__ == "__main__":
    # 测试神经网络
    print("测试神经网络模块...")
    
    # 创建网络
    nn = NeuralNetwork(input_size=10, hidden_sizes=[64, 32], output_size=5)
    print(f"参数数量: {nn.get_param_count()}")
    
    # 设置随机参数
    params = create_random_params(nn.get_param_count())
    nn.set_params(params)
    
    # 测试前向传播
    test_input = np.random.randn(10)
    output = nn.predict(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 测试保存和加载
    save_network(nn, "test_model.npy")
    loaded_nn = load_network("test_model.npy")
    
    # 验证加载的模型
    loaded_output = loaded_nn.predict(test_input)
    assert np.allclose(output, loaded_output), "加载的模型输出不一致！"
    print("保存和加载测试通过！")
    
    # 清理测试文件
    import os
    os.remove("test_model.npy")
    print("测试完成！")

