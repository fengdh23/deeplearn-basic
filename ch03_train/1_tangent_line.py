import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_diff

# 原函数 y = 0.01x^2 + 0.1x
def f(x):
    return 0.01 * x**2 + 0.1 * x

# 切线方程函数，返回切线函数 y = ax + b
def tangent_line(f, x):
    y = f(x)
    # 计算x处切线的斜率（利用数值微分计算x处的导数）
    a = numerical_diff(f, x) # 斜率 a
    print("切线斜率为：", a) # 0.1999999999990898
    # 根据切线过(x, y)点，计算截距
    b = y - a * x # 截距 b ∂f/(∂x_i ) (a_1,a_2,…,a_n )=lim┬(∆x_i→0)⁡〖(f(a_1,…a_i+∆x_i,…,a_n )-f(a_1,…a_i,…,a_n ))/(∆x_i )〗
    # return a, b
    # return lambda x: a * x + b
    return lambda z: a * z + b

# 定义画图范围
x = np.arange(0.0, 20.0, 0.1)
y = f(x)

# 计算x=5处的切线方程
f_line = tangent_line(f, x=5)
y_line = f_line(x)

plt.plot(x, y)  # 原函数曲线
plt.plot(x, y_line) # 切线
plt.show()

