# 阶跃函数 一个标量
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int) # 返回一个布尔值，转换为整型

# Sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ReLU函数
def relu(x):
    return np.maximum(0, x)

# Softmax函数
def softmax0(x):
    return np.exp(x) / np.sum(np.exp(x))

# 考虑输入可能是矩阵的情况
def softmax(x):
    # 如果是二维矩阵
    if x.ndim  == 2:
        x = x.T # 转置 之前是按行取最大，现在是按列取最大
        x = x - np.max(x, axis=0) # axis 由1 改为0
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 溢出处理策略 指数如果很大，10000, 数值会非常大，会溢出
    x = x - np.max(x) # 广播，最大值是0 ，其他都是负值
    return np.exp(x) / np.sum(np.exp(x))

# 恒等函数
def identity(x):
    return x

# 损失函数
# MSE
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2) # 向量相减

# 交叉熵误差
def cross_entropy(y, t):
    # 将y转为二维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 将t转换为顺序编码（类别标签）
    if t.size == y.size:
        t = t.argmax(axis=1)
    n = y.shape[0]
    return -np.sum( np.log(y[np.arange(n), t] + 1e-10) ) / n

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    print("step_function: ",step_function(x))
    print('sigmoid: ',sigmoid(x))
    print('tanh: ',np.tanh(x))
    print('relu: ',relu(x))

    X = np.array([[0,1,2], [3,4,5], [6,7,8], [-1,-2,-3]])
    print('softmax: ',softmax(X))