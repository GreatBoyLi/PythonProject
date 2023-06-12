import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from liwp import litorch as li
from matplotlib import pyplot as plt


d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.299, 0.244, 0.255])
train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
                                             torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.ToTensor(), normalize])
test_augs = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor(), normalize])

# pretrained=True代表不光把模型拿过来，也把训练好的parameters拿过来
pretrained_net = torchvision.models.resnet18(pretrained=False)

# 全连接层，最后那个输出层
print(pretrained_net.fc)

# 输入的还是之前的，但是输出的变成了2，因为只有2个类
pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 2)
# nn.init.xavier_uniform_(pretrained_net.fc.weight)


def train_fine_tuning(net, lr, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder
                                             (os.path.join(data_dir, 'train'), transform=train_augs),
                                             batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder
                                            (os.path.join(data_dir, 'test'), transform=test_augs),
                                            batch_size=batch_size, shuffle=True)
    devices = li.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        param_lx = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': param_lx},
                                   {'params': net.fc.parameters(), 'lr': lr * 10}],
                                  lr=lr, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)

    li.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


train_fine_tuning(pretrained_net, 5e-4, param_group=False)
