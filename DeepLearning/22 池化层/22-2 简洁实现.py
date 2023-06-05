import torch
from torch import nn
from liwp import litorch as li


X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))

# 深度学习框架中的步幅与池化窗口的大小相同
pool2d = nn.MaxPool2d(3)
print(pool2d(X))

# 手动指定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
print(pool2d(X))

# 在每个通道上池化
X = torch.cat((X, X + 1), 1)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
