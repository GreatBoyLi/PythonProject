import torch
from torch import nn
from d2l import torch as d2l
import liwp.litorch as liwp


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # PyTorch不会隐式地调整输入的形状
    # 因此，我们定义也展平层(flatten)在线性层前调整网格输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    liwp.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
