import numpy as np

from common.functions import softmax, cross_entropy
from common.gradient import numerical_gradient

# 定义一个简单神经网络类
class SimpleNet:
    # 初始化
    def __init__(self):
        self.W = np.random.randn(2, 3) # 2 * 3 矩阵
    # 前向传播
    def forward(self, X):
        a = X @ self.W # @ 表示矩阵乘法运算
        y = softmax(a)
        return y
    # 计算损失值
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy(y, t)
        return loss

# 主流程
if __name__ == '__main__':
    # 1. 定义数据
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1]) # 目标值，分类标签
    # 2. 定义神经网络模型
    net = SimpleNet()
    # 3. 计算梯度
    f = lambda _: net.loss(x, t) # W 全局变量
    gradw = numerical_gradient(f, net.W)

    print(gradw)
    # [[0.53017906  0.03412196 - 0.56430102]
    #  [0.79526859  0.05118294 - 0.84645154]]