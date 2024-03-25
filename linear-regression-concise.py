import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# print(features)
# print(labels)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型,我们将两个参数传递到nn.Linear中。 
# 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 计算均方误差使用的是MSELoss类，也称为平方范数。 默认情况下，它返回所有样本损失的平均值
loss = nn.MSELoss()
# SGD:随机梯度下降优化器类,小批量随机梯度下降只需要设置lr值，这里设置为0.03,
# net.parameters()会返回网络中所有需要梯度更新的参数（这包括由 nn.Linear 层创建的所有权重和偏置）。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#计算每个迭代周期后的损失，并打印它来监控训练过程。

for epoch in range(3):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:', true_b - b)