import torch
from torch import nn
from liwp import litorch as li


x = torch.tensor([1, 2, 3])
print(x.device)

# 储存在GPU
X = torch.ones(2, 3, device=li.try_gpu())
print(X)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=li.try_gpu())

print(net(X))

# 确认模型参数存储在GPU上
print(net[0].weight.data.device)
