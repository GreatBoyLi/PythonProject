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

