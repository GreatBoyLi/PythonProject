from liwp import litorch as li
import torchvision
from torch import nn
import torch
from d2l import torch as d2l


batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = li.load_data_voc(batch_size, crop_size)

devices = li.try_all_gpus()
pretrained_net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
num_classes = 21
mid = 256
net.add_module('final_conv', nn.Conv2d(512, mid, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(mid, num_classes, kernel_size=64, padding=16, stride=32))
net.load_state_dict(torch.load('111.params'))
net.to(devices[0])


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(li.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]


# voc_dir = li.download_extract('voc2012,' 'VOCdevkit/VOC2012')
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = li.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1, 2, 0), pred.cpu(), torchvision.transforms.functional.crop(
        test_labels[i], *crop_rect).permute(1, 2, 0)]

li.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
