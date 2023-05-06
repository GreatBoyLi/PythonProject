import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import d2l.torch as d2l

num_workers = 0


def loadDataFashionMnist(batch_size: int, resize=None):
    """
    下载Fashion-MNIST数据集，然后将其加载到内存中
    :param batch_size: 读取批次的大小
    :param resize: 将图像数据重新调整大小
    :return:
    """
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    mnistTrain = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnistTest = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnistTrain, batch_size, shuffle=True, num_workers=getDataLoaderWorkers()),
            torch.utils.data.DataLoader(mnistTest, batch_size, shuffle=True, num_workers=getDataLoaderWorkers()))


def getDataLoaderWorkers():
    """
    获取torch.utils.data.DataLoader的子进程数，默认为0
    :return: 返回子进程数
    """
    return num_workers


def setDataLoaderWorkers(loadWorkers=0):
    global num_workers
    num_workers = loadWorkers


def softmax(X):
    """
    对X进行softmax操作
    :param X:
    :return: 返回对应的结果
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        plt.show()

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear',
                 yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
        # 使用lambda函数捕获参数
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.show()
        d2l.display.display(self.fig)
        d2l.display.clear_output(wait=True)


def train_epoch_ch3(net, train_iter, loss, updater):
    """
    Softmax的回归训练
    :param net:
    :param train_iter:
    :param loss:
    :param updater:
    :return:
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def accuracy(y_hat, y):
    """
    计算预测正确的数量
    :param y_hat:
    :param y:
    :return:
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        # if train_loss < 0.7:
        #     return
        # assert train_loss < 0.5, train_loss
        # assert 1 >= train_acc > 0.7, train_acc
        # assert 1 >= test_acc > 0.7, test_acc


def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度
    :param net:
    :param data_iter:
    :return:
    """
    if isinstance(net, torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    print(titles)
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.subplots_adjust(wspace=1.0)
    plt.show()