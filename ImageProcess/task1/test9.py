import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from skimage import io


carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
# encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)
# encrypt2 = io.imread('./image/encrypt/encrypt2.pgm', as_gray=True)
# encrypt3 = io.imread('./image/encrypt/encrypt3.pgm', as_gray=True)


# 傅里叶变换
carrier_fft = fft.fft2(carrier)
fft_img = 20 * np.log(np.abs(carrier_fft))
plt.subplot(221)
plt.imshow(fft_img)

carrier_fft[256 - 50: 256 + 50, 256 - 50: 256 + 50] = 0
plt.subplot(222)
plt.imshow(20 * np.log(np.abs(carrier_fft)))

carrier_fft_ifft = fft.ifft2(carrier_fft)

carrier_fft1 = fft.fft2(carrier_fft_ifft)

plt.subplot(223)
plt.imshow(20 * np.log(np.abs(carrier_fft1)))

carrier_fft_ifft_img = np.abs(carrier_fft_ifft)
carrier_fft_ifft_img_fft = fft.ifft2(carrier_fft_ifft_img)
plt.subplot(224)
plt.imshow(20 * np.log(np.abs(carrier_fft_ifft_img_fft)))

count = 0
for x in range(512):
    for y in range(512):
        if carrier_fft[x][y] == carrier_fft1[x][y]:
            count += 1

print(count)

plt.show()
