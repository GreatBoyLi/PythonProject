import os
import torch
import torchvision
from d2l import torch as d2l
from liwp import litorch as li
import matplotlib.pyplot as plt


# 下载数据集并解压
# d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar', '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')


# 将所有输入的图像和标签读入内存，voc格式
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels


def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = li.voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [img for img in imgs if(img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return feature, li.voc_label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir), batch_size, shuffle=True,
                                             drop_last=True)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir), batch_size, shuffle=True,
                                             drop_last=True)
    return train_iter, test_iter




train_features, train_labels = read_voc_images(voc_dir, True)

n = 5
# imgs = train_features[0:n] + train_labels[0:n]
# # 训练时channel放在前面，显示时channel放在后面
# imgs = [img.permute(1, 2, 0) for img in imgs]
# li.show_images(imgs, 2, n,scale=3)

# voc_colormap2label = li.voc_colormap2label()
# y = li.voc_label_indices(train_labels[0],voc_colormap2label)
# print(y[105:115, 130:140], li.VOC_CLASSES[1])

# imgs =[]
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
# imgs = [img.permute(1, 2, 0) for img in imgs]
# li.show_images(imgs[::2] + imgs[1::2], 2, n)

crop_size = (320 ,480)
voc_train = VOCSegDataset(True, crop_size,voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
batch_size = 1
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True)
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
