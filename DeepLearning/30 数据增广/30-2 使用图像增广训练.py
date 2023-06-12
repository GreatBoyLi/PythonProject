import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from liwp import litorch as li
import time


all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)

d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

plt.show()

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=augs, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return data_loader


def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = li.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    timer, num_batchs = d2l.Timer(), len(train_iter)
    # 使用多GPU进行模型训练
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        start = time.time()
        # 四个维度：储存训练损失，训练准确度，实例数，特点数
        metric = li.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
        end = time.time()
        test_acc = li.evaluate_accuracy_gpu(net, test_iter)
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[3]
        print(
            f'第{epoch + 1}轮，花费时间为{end - start:.3f}秒。训练损失是{train_l:.3f},'
            f'训练数据预测精度是{train_acc},测试数据预测精确度是{test_acc}。')
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


batch_size, devices, net = 256, li.try_all_gpus(), d2l.resnet18(10, 3)


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_with_data_aug(train_augs, test_augs, net)

