import torch
import torchvision
import torch.utils.data

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

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
