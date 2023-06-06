import torch
from torch import nn
from skimage import io
import cv2


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


clone = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

clone.load_state_dict(torch.load('LeNet.params'))

while(1):
    name = input("输入图片的路径和名称：")
    if name == '':
        print(1)
        break
    else:
        print(name)
        picture = io.imread(name, as_gray=True)
        dest = cv2.resize(picture, dsize=(28, 28)).reshape(1, 1, 28, 28)
        y = clone(torch.tensor(dest, dtype=torch.float32))
        print(y)
        print('****************************************************************')
