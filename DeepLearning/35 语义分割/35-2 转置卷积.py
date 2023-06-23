import torch
from torch import nn
from d2l import torch as d2l


# 基于的转置卷积的运算
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

X, K = X.reshape(1, 1, 2, 2,), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 有填充，结果会变小，是说结果是经过填充得到的，最终要剪去padding=1这个填充
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

# 步幅增加，结果会变大，因为相加的步幅增大了
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 多通道
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10 ,20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)

print('*****************************************************************')

# 卷积变矩阵乘法
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(Y)


# 3x3的图像，kernel=2
def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W


W = kernel2matrix(K)
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))
