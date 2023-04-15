import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 使用框架中现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# nn是神经网络的缩写
net = nn.Sequential(nn.Linear(2, 1))

# 初始化W和b的值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 均方误差
loss = nn.MSELoss()

# 随机梯度下降算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 迭代3个周期
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

print(f'w的估计误差：{true_w - net[0].weight}')
print(f'b的估计误差：{true_b - net[0].bias}')
