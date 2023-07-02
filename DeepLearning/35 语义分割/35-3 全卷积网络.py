import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
from liwp import litorch as li


# 初始化转置卷积层，双线性插值的一个核
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


pretrained_net = torchvision.models.resnet18(pretrained=True)
# 倒数三层
# print(list(pretrained_net.children())[-3:])

net = nn.Sequential(*list(pretrained_net.children())[:-2])
# X = torch.rand(size=(1, 3, 320, 480))
# print(net(X).shape)

# 使用1x1卷积层将输出通道转换为Pascal VOC2012数据集的类数(21类)。将要素地图的高度和宽度增加32倍
num_classes = 21
mid = 256
net.add_module('final_conv', nn.Conv2d(512, mid, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(mid, num_classes, kernel_size=64, padding=16, stride=32))

# 双线性插值测试
# conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
# conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
# img = torchvision.transforms.ToTensor()(d2l.Image.open('../images/catdog.jpg'))
# X = img.unsqueeze(0)
# Y = conv_trans(X)
# out_img = Y[0].permute(1, 2, 0).detach()
# print('input image shape:', img.permute(1, 2, 0).shape)
# plt.imshow(img.permute(1, 2, 0))
# plt.show()
# print('output image shape:', out_img.shape)
# plt.imshow(out_img)
# plt.show()

W = bilinear_kernel(mid, num_classes, kernel_size=64)
net.transpose_conv.weight.data.copy_(W)

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = li.load_data_voc(batch_size, crop_size)


def loss(inputs, tartgets):
    return F.cross_entropy(inputs, tartgets, reduction='none').mean(1).mean(1)


num_epochs, lr, wd, device = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
li.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs)

torch.save(net.state_dict(), '111.params')
