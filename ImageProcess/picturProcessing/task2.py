import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/2.jpg', cv2.IMREAD_GRAYSCALE)

# 统计灰度级分布
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# 计算累积分布函数
cdf = hist.cumsum()
cdf_normalized = cdf / cdf.max()

# 直方图均衡化
img_equalized = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        b = img[i, j]
        a = cdf_normalized[b] 
        c = a * 255
        img_equalized[i,j] = c

# 统计均衡化后的灰度级分布
hist_equalized, bins_equalized = np.histogram(img_equalized.flatten(), 256, [0, 256])

# 将灰度级范围归一化到[0, 1]
img_normalized = img / 255.0
img_equalized_normalized = img_equalized / 255.0

# 绘制直方图
plt.figure(figsize=(10, 6.5))
plt.subplot(221)
plt.title('Histogram of Input Image')
plt.hist(img_normalized.flatten(), bins=256, range=(0,1))
plt.subplot(222)
plt.title('Histogram of Equalized Image')
plt.hist(img_equalized_normalized.flatten(), bins=256, range=(0,1))
plt.subplot(223)
plt.imshow(img_normalized, cmap='gray')
plt.title('Input Image')
plt.subplot(224)
plt.imshow(img_equalized_normalized, cmap='gray')
plt.title('Equalized Image')
plt.show()