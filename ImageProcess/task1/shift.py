import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from skimage import io


carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)
encrypt2 = io.imread('./image/encrypt/encrypt2.pgm', as_gray=True)
encrypt3 = io.imread('./image/encrypt/encrypt3.pgm', as_gray=True)


# 傅里叶变换
carrier_fft = fft.fftshift(fft.fft2(carrier))
encrypt_fft1 = fft.fftshift(fft.fft2(encrypt1))
encrypt_fft2 = fft.fftshift(fft.fft2(encrypt2))
encrypt_fft3 = fft.fftshift(fft.fft2(encrypt3))

carrier_fft_shift = fft.fftshift(carrier_fft)
plt.subplot(221)
plt.imshow(20 * np.log(np.abs(carrier_fft_shift)))

# 装载第一张图片
carrier_fft[0: 128, 0: 128] = encrypt_fft1  [128: 256, 128: 256] / 500
carrier_fft[0: 128, 128: 256] = encrypt_fft1[256: 384, 128: 256] / 500
carrier_fft[0: 128, 256: 384] = encrypt_fft1[128: 256, 256: 384] / 500
carrier_fft[0: 128, 384: 512] = encrypt_fft1[256: 384, 256: 384] / 500
# # 装载第二张图片
# carrier_fft[256: 384, 128: 256] = encrypt_fft2[0: 128, 0: 128] / 500
# carrier_fft[384: 512, 128: 256] = encrypt_fft2[384: 512, 0: 128] / 500
# carrier_fft[0: 128, 256: 384] = encrypt_fft2[0: 128, 384: 512] / 500
# carrier_fft[128: 256, 256: 384] = encrypt_fft2[384: 512, 384: 512] / 500
# # 装载第三张图片
# carrier_fft[256: 384, 256: 384] = encrypt_fft3[0: 128, 0: 128] / 500
# carrier_fft[384: 512, 256: 384] = encrypt_fft3[384: 512, 0: 128] / 500
# carrier_fft[128: 256, 384: 512] = encrypt_fft3[0: 128, 384: 512] / 500
# carrier_fft[256: 384, 384: 512] = encrypt_fft3[384: 512, 384: 512] / 500

# 加密过后的载体图像傅里叶频谱
carrier_fft = fft.ifftshift(carrier_fft)
plt.subplot(222)
plt.imshow(20 * np.log(np.abs(carrier_fft)))

carrier_fft_ifft = fft.ifft2(carrier_fft)
plt.subplot(223)
plt.imshow(np.abs(carrier_fft_ifft), cmap='gray')

carrier_fft_ifft = np.abs(carrier_fft_ifft)
carrier_fft2 = fft.fft2(carrier_fft_ifft)
carrier_fft2 = fft.fftshift(carrier_fft2)
plt.subplot(224)
plt.imshow(20 * np.log(np.abs(carrier_fft2)))


a = np.zeros((512, 512))
__encrypt_fft1 = a + a * 1j
__encrypt_fft2 = a + a * 1j
__encrypt_fft3 = a + a * 1j
__proto_image = a + a * 1j

# 拆解第一张图片
__encrypt_fft1[128: 256, 128: 256] = carrier_fft2[0: 128, 0: 128] * 500
__encrypt_fft1[256: 384, 128: 256] = carrier_fft2[0: 128, 128: 256] * 500
__encrypt_fft1[128: 256, 256: 384] = carrier_fft2[0: 128, 256: 384] * 500
__encrypt_fft1[256: 384, 256: 384] = carrier_fft2[0: 128, 384: 512] * 500

__encrypt_fft1_ifft = fft.ifft2(__encrypt_fft1)
plt.subplot(223)
plt.imshow(np.abs(__encrypt_fft1_ifft), cmap='gray')
# 拆解第二张图片
__encrypt_fft2[0: 128, 0: 128] = carrier_fft2[256: 384, 128: 256] * 500
__encrypt_fft2[384: 512, 0: 128] = carrier_fft2[384: 512, 128: 256] * 500
__encrypt_fft2[0: 128, 384: 512] = carrier_fft2[0: 128, 256: 384] * 500
__encrypt_fft2[384: 512, 384: 512] = carrier_fft2[128: 256, 256: 384] * 500
# 拆解第三张图片
__encrypt_fft3[0: 128, 0: 128] = carrier_fft2[256: 384, 256: 384] * 500
__encrypt_fft3[384: 512, 0: 128] = carrier_fft2[384: 512, 256: 384] * 500
__encrypt_fft3[0: 128, 384: 512] = carrier_fft2[128: 256, 384: 512] * 500
__encrypt_fft3[384: 512, 384: 512] = carrier_fft2[256: 384, 384: 512] * 500


__proto_image[0: 128, 0: 128] = carrier_fft2[0: 128, 0: 128]
__proto_image[384: 512, 0: 128] = carrier_fft2[384: 512, 0: 128]
__proto_image[0: 128, 384: 512] = carrier_fft2[0: 128, 384: 512]
__proto_image[384: 512, 384: 512] = carrier_fft2[384: 512, 384: 512]


plt.show()
