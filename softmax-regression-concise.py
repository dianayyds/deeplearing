import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 在线性层前定义展平层（flatten），来调整网络输入的形状,特征是28*28=784,类别
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights) #遍历所有层执行初始化

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.plt.show()