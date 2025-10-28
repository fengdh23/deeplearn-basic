import numpy as np
import pandas as pd
import joblib # 高效地保存和加载Python对象，常用于机器学习模型的序列
from sklearn.model_selection import train_test_split #用于将数据集划分为训练集和测试集
from sklearn.preprocessing import MinMaxScaler # 用于将特征数据缩放到指定范围（通常是0-1之间）像素范围：0~255
from common.functions import sigmoid, softmax   # 激活函数

# 读取数据
def get_data():
    # 1. 从文件加载数据集 783个像素点 10个标签
    data = pd.read_csv("../data/train.csv")
    # 2. 划分数据集 从原始数据中删除"label"列，将剩余的所有特征列作为输入特征X。其中axis=1表示按列操作，drop()方法会返回一个新的DataFrame，
    # 不包含指定的"label"列，从而实现特征与标签的分离
    X = data.drop("label", axis=1)
    y = data["label"]
    # 数据按7:3比例分割成训练集和测试集，其中test_size=0.3表示测试集占30%，
    # random_state=42(常用的"魔法数字"，在机器学习社区中被广泛采用)确保每次运行结果一致
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 3. 特征工程：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train) # 训练集fit
    x_test = scaler.transform(x_test) # 测试集只需要转换

    return x_test, y_test # 返回测试集和标签  x_train 已经训练过了，目前用不到

# 初始化神经网络
def init_network():
    # 直接从文件中加载字典对象 即加载模型
    network = joblib.load("../data/nn_sample")
    return network

# 前向传播
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层进行计算传递
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1) # 隐藏层激活函数
    # 隐藏层
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    # 输出层
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3) # 输出层激活函数

    return y

# 主流程
# 1. 获取测试数据 x 是测试数据，y 是标签
x, y = get_data()
# print(x.shape) # (12600, 784)
# print(y.shape) # (12600,)

# 2. 创建模型（加载参数）
network = init_network()
print(network['W1'].shape)# (784, 50)
print(network['W2'].shape)# (50, 100)
print(network['W3'].shape)# (100, 10)
print(network['b1'].shape)# (50,)
print(network['b2'].shape)# (100,)
print(network['b3'].shape)# (10,)

# 3. 前向传播（测试）
y_proba = forward(network, x)
print(y_proba.shape) # (12600, 10)

# 4. 将分类概率转换为分类标签 才能和原始的标签进行比较
y_pred = np.argmax(y_proba, axis=1) # 预测概率矩阵 y_proba 中每行的最大概率值对应的列索引作为预测结果。

# 5. 计算分类准确率 相等则加1
accuracy_cnt = np.sum(y_pred == y) # true --> 1
n = x.shape[0] # 数据的总个数 12600
print("Accuracy: ", accuracy_cnt / n)