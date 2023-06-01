import torch
from torch import nn


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand((2, 4))
print(net(X))
print("*********************")
print(net[2].state_dict())
print("*********************")
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# 参数是复合的对象，包含值、梯度和额外的信息
print(net[2].weight.grad == None)
print("*********************")
# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param) for name, param in net.named_parameters()])
print("*********************")
print(net.state_dict()['2.bias'].data)
print("*********************")


def block1():
    return nn.Sequential(nn.Linear(4, 8),nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)
print("*********************")
print(rgnet[0][1][0].bias.data)
print("*********************")


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
print("*********************")


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
print("*********************")


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
print("*********************")


# 参数绑定。给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])

