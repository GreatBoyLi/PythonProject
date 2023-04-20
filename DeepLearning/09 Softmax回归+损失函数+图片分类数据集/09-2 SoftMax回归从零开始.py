import torch
from d2l import torch as d2l
import liwp.litorch as liwp


def net(X, W1, b1):
    """
    定义softmax回归模型
    :param X: 数据集
    :param W1: 权重
    :param b1: 偏置
    :return:
    """
    return liwp.softmax(torch.matmul(X.reshape(-1, W.shape[0]), W1) + b1)


def cross_entropy(y_hat, y):
    """
    实现交叉熵损失
    :param y_hat:
    :param y:
    :return:
    """
    return -torch.log(y_hat[range(len(y_hat)), y])


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
    metric = liwp.Accumulator(2)
    with torch.no_grad:
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = liwp.loadDataFashionMnist(batch_size)
    # 28 * 28 = 784
    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
