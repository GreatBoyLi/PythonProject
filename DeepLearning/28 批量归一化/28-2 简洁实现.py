import torch
from torch import nn
from liwp import litorch as li


net = torch.nn.Sequential(
    li.Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 2.0, 10, 256
train_iter, test_iter = li.loadDataMnist(batch_size)
li.train_ch6(net, train_iter, test_iter, num_epochs, lr, li.try_gpu())
