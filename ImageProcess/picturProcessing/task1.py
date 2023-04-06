from skimage import io, feature, transform
from matplotlib import pyplot as plt
import numpy as np


def cimageprocess(img, x1: int, x2: int, y1: int, y2: int):
    xsize, ysize = img.shape
    img_new = np.zeros((xsize, ysize))
    if x1 > x2:
        print("x1不能大于x2")
    for i in range(0, xsize):
        for j in range(0, ysize):
            if img[i][j] <= x1:
                img_new[i][j] = y1 * img[i][j] / x1
            elif img[i][j] <= x2:
                img_new[i][j] = (y2 - y1) * (img[i][j] - x1) / (x2 - x1) + y1
            else:
                img_new[i][j] = (255 - y2) * (img[i][j] - x2) / (255 - x2) + y2

    return img_new



img = io.imread('.\images\coin.jpg', as_gray=True) * 255
fig, axes = plt.subplots(1, 3)
image1 = cimageprocess(img, 0.3 * 255, 0.7 * 255, 0.15 * 255, 0.85 * 255)
image2 = cimageprocess(img, 0.15 * 255, 0.85 * 255, 0.3 * 255, 0.7 * 255)
axe = axes.ravel()
axe[0].imshow(img, cmap="gray")
axe[1].imshow(image1, cmap="gray")
axe[2].imshow(image2, cmap="gray")

plt.show()
