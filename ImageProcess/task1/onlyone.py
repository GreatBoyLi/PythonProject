from skimage import io
from numpy import fft
import numpy as np
from matplotlib import pyplot as plt

one = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
plt.subplot(221)
plt.imshow(one, cmap='gray')

one_fft = fft.fft2(one)
plt.subplot(222)
plt.imshow(20 * np.log(np.abs(one_fft)))

one_fft_ifft = fft.ifft2(one_fft)

one_fft_ifft_img = (np.abs(one_fft_ifft))

count = 0
for x in range(512):
    for y in range(512):
        if one[x][y] == one_fft_ifft_img[x][y]:
            count += 1

print(count)

plt.subplot(223)
plt.imshow(one_fft_ifft_img, cmap='gray')


new_one_fft = fft.fft2(one_fft_ifft_img)
plt.subplot(224)
plt.imshow(20 * np.log(np.abs(new_one_fft)))
count1 = 0
for x in range(512):
    for y in range(512):
        if one_fft[x][y] == new_one_fft[x][y]:
            count1 += 1

print(count1)
plt.show()
