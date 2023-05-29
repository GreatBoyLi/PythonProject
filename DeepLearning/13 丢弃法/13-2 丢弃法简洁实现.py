import torch.nn as nn
import torch
import liwp.litorch as liwp


dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    # 在第一个全连接层加一个丢弃层
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    # 在第二个全连接层之后添加一个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


num_epochs, lr, batch_size = 10, 0.5, 256

net.apply(init_weights)
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
train_iter, test_iter = liwp.loadDataFashionMnist(batch_size)
loss = nn.CrossEntropyLoss()
liwp.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
