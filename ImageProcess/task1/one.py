from numpy import fft
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)

carrier_fft = fft.fft2(carrier)
encrypt_fft1 = fft.fft2(encrypt1)

# 装载第一张图片
carrier_fft[169: 341, 0: 256] = encrypt_fft1[0: 172, 0: 256] / 500
carrier_fft[0: 169, 169: 341] = encrypt_fft1[343: 512, 0: 172] / 500
carrier_fft[169: 341, 256: 512] = encrypt_fft1[0: 172, 256: 512] / 500
carrier_fft[341: 512, 169: 341] = encrypt_fft1[341: 512, 340: 512] / 500

# 加密过后的载体图像傅里叶频谱
fftImg1 = 20 * np.log(np.abs(fft.fftshift(carrier_fft)))
plt.subplot(221)
plt.imshow(fftImg1)

carrier_fft_ifft = fft.ifft2(carrier_fft)
carrier_fft_ifft_img = np.abs(carrier_fft_ifft)
plt.subplot(222)
plt.imshow(np.abs(carrier_fft_ifft), cmap='gray')

carrier_fft_ifft_img_fft = fft.fft2(carrier_fft_ifft_img)

a = np.zeros((512, 512))
__encrypt_fft1 = a + a * 1j
__proto_image = a + a * 1j

# 拆解第一张图片
__encrypt_fft1[0: 172, 0: 256] = carrier_fft_ifft_img_fft[169: 341, 0: 256] * 500
__encrypt_fft1[343: 512, 0: 172] = carrier_fft_ifft_img_fft[0: 169, 169: 341] * 500
__encrypt_fft1[0: 172, 256: 512] = carrier_fft_ifft_img_fft[169: 341, 256: 512] * 500
__encrypt_fft1[341: 512, 340: 512] = carrier_fft_ifft_img_fft[341: 512, 169: 341] * 500

__proto_image[0: 169, 0: 169] = carrier_fft_ifft_img_fft[0: 169, 0: 169]
__proto_image[341: 512, 0: 169] = carrier_fft_ifft_img_fft[341: 512, 0: 169]
__proto_image[0: 169, 341: 512] = carrier_fft_ifft_img_fft[0: 169, 341: 512]
__proto_image[341: 512, 341: 512] = carrier_fft_ifft_img_fft[341: 512, 341: 512]

proto_img_ifft = fft.ifft2(__proto_image)
proto_img_ifft_img = np.abs(proto_img_ifft)

encrypt1_ifft = fft.ifft2(__encrypt_fft1)
fftImg3 = 20 * np.log(np.abs(carrier_fft_ifft_img_fft))
fftImg2 = 20 * np.log(np.abs(__encrypt_fft1))
plt.subplot(223)
plt.imshow(fftImg3)

encrypt1_img = np.abs(encrypt1_ifft)
plt.subplot(224)
plt.imshow(encrypt1_img, cmap='gray')

plt.show()
