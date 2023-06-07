import torchvision
import torch.utils.data as data
from d2l import torch as d2l
from torchvision import transforms
from torch import nn
import torch
from liwp import litorch as li


data_root = "./data"
trans = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root=data_root, train=True, transform=trans, download=True)
test_set = torchvision.datasets.MNIST(root=data_root, train=False, transform=trans, download=True)

train_iter = data.DataLoader(train_set, batch_size=256, shuffle=True)
test_iter = data.DataLoader(test_set, shuffle=True, batch_size=256)

net = torch.nn.Sequential(
    li.Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs = 0.5, 10

li.train_ch6(net, train_iter, test_iter, num_epochs, lr, li.try_gpu())
torch.save(net.state_dict(), 'LeNet.params')
