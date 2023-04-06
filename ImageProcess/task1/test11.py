from numpy import fft
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)

carrier_fft = fft.fft2(carrier)
encrypt_fft1 = fft.fft2(encrypt1)

# 装载第一张图片
carrier_fft[128: 256, 0: 128] = encrypt_fft1[0: 128, 0: 128] / 500
carrier_fft[256: 384, 0: 128] = encrypt_fft1[384: 512, 0: 128] / 500
carrier_fft[0: 128, 128: 256] = encrypt_fft1[0: 128, 384: 512] / 500
carrier_fft[128: 256, 128: 256] = encrypt_fft1[384: 512, 384: 512] / 500

# 装载第二张图片
carrier_fft[256: 384, 128: 256] = encrypt_fft1[0: 128, 0: 128] / 500
carrier_fft[384: 512, 128: 256] = encrypt_fft1[384: 512, 0: 128] / 500
carrier_fft[0: 128, 256: 384] = encrypt_fft1[0: 128, 384: 512] / 500
carrier_fft[128: 256, 256: 384] = encrypt_fft1[384: 512, 384: 512] / 500
# 装载第三张图片
carrier_fft[256: 384, 256: 384] = encrypt_fft1[0: 128, 0: 128] / 500
carrier_fft[384: 512, 256: 384] = encrypt_fft1[384: 512, 0: 128] / 500
carrier_fft[128: 256, 384: 512] = encrypt_fft1[0: 128, 384: 512] / 500
carrier_fft[256: 384, 384: 512] = encrypt_fft1[384: 512, 384: 512] / 500

# 加密过后的载体图像傅里叶频谱
fftImg1 = 20 * np.log(np.abs(carrier_fft))
plt.subplot(221)
plt.imshow(fftImg1)

carrier_fft_ifft = fft.ifft2(carrier_fft)
carrier_fft_ifft_img = np.abs(carrier_fft_ifft)
plt.subplot(222)
plt.imshow(np.abs(carrier_fft_ifft), cmap='gray')

carrier_fft_ifft_img_fft = fft.fft2(carrier_fft_ifft_img)

a = np.zeros((512, 512))
__encrypt_fft1 = a + a * 1j

# 拆解第一张图片
__encrypt_fft1[0: 128, 0: 128] = carrier_fft_ifft_img_fft[128: 256, 0: 128] * 500
__encrypt_fft1[384: 512, 0: 128] = carrier_fft_ifft_img_fft[256: 384, 0: 128] * 500
__encrypt_fft1[0: 128, 384: 512] = carrier_fft_ifft_img_fft[0: 128, 128: 256] * 500
__encrypt_fft1[384: 512, 384: 512] = carrier_fft_ifft_img_fft[128: 256, 128: 256] * 500

encrypt1_ifft = fft.ifft2(__encrypt_fft1)
fftImg2 = 20 * np.log(np.abs(__encrypt_fft1))
plt.subplot(223)
plt.imshow(fftImg2)

encrypt1_img = np.abs(encrypt1_ifft)
plt.subplot(224)
plt.imshow(encrypt1_img, cmap='gray')

plt.show()
