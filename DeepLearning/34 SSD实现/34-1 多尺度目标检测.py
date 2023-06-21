import torch
from matplotlib import pyplot as plt
from liwp import litorch as li


img = plt.imread('../images/catdog.jpg')
h, w = img.shape[:2]
print(h, w)


def display_anchors(fmap_w, fmap_h, s):
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = li.multibox_prior(fmap, sizes=s, ratios=[0.5, 2])
    bbox_scale = torch.tensor((w, h, w, h))
    li.show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)
    plt.show()


display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
display_anchors(fmap_w=4, fmap_h=4, s=[0.15, 0.1])
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

