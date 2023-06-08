import torch
from torch import nn
from liwp import litorch as li


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append((nn.ReLU()))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1
    for(num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blocks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 10)
    )


# net = vgg(conv_arch)
# X = torch.randn(size=(1, 1, 224, 224))
# for block in net:
#     X = block(X)
#     print(block.__class__.__name__, 'output shape:\t', X.shape)

# 由于VGG-11比AlexNet计算量更大，因此我们柳妈了一个通道数较少的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = li.loadDataMnist(batch_size, 224)
li.train_ch6(net, train_iter, test_iter, num_epochs, lr, li.try_gpu())
torch.save(net.state_dict(), 'VGG.params')
