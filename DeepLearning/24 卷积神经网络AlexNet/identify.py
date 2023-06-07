import torch
from torch import nn
from skimage import io
import cv2
from liwp import litorch as li




clone = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

clone.load_state_dict(torch.load('AlexNet.params'))

while(1):
    name = input("输入图片的路径和名称：")
    if name == '':
        print(1)
        break
    else:
        print(name)
        picture = io.imread(name, as_gray=True)
        dest = cv2.resize(picture, dsize=(224, 224)).reshape(1, 1, 224, 224)
        y = clone(torch.tensor(dest, dtype=torch.float32))
        print(y)
        print(y.argmax(axis=1))
        print('****************************************************************')
