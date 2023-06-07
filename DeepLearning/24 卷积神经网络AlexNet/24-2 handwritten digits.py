import torchvision
import torch.utils.data as data
from d2l import torch as d2l
from torchvision import transforms
from torch import nn
import torch
from liwp import litorch as li
import cv2
import numpy as np


data_root = "../23 卷积神经网络LeNet/data"
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(224))
trans = transforms.Compose(trans)
train_set = torchvision.datasets.MNIST(root=data_root, train=True, transform=trans, download=True)
test_set = torchvision.datasets.MNIST(root=data_root, train=False, transform=trans, download=True)

train_iter = data.DataLoader(train_set, batch_size=128, shuffle=True)
test_iter = data.DataLoader(test_set, shuffle=True, batch_size=128)

X, y = next(iter(data.DataLoader(test_set, batch_size=18)))
d2l.show_images(X.reshape(18, 224, 224), 2, 9)
d2l.plt.show()
b = zip(enumerate(X.reshape(18, 224, 224)), y)
for item, label in zip(enumerate(X.reshape(18, 224, 224)), y):
        i, image = item
        image = (image.numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"test{label}.jpg", image)

#
# net = nn.Sequential(
#         nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
#         nn.MaxPool2d(kernel_size=3, stride=2),
#         nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
#         nn.MaxPool2d(kernel_size=3, stride=2),
#         nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
#         nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
#         nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
#         nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
#         nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5),
#         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
#         nn.Linear(4096, 10)
#     )
#
# lr, num_epochs = 0.01, 10
# li.train_ch6(net, train_iter, test_iter, num_epochs, lr, li.try_gpu())
# torch.save(net.state_dict(), 'AlexNet.params')
