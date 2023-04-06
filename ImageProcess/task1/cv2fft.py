import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from skimage import io
import cv2


carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)
encrypt2 = io.imread('./image/encrypt/encrypt2.pgm', as_gray=True)
encrypt3 = io.imread('./image/encrypt/encrypt3.pgm', as_gray=True)

# 傅里叶变换
carrier_fft = cv2.dft(np.float32(carrier), flags=cv2.DFT_COMPLEX_OUTPUT)
encrypt_fft1 = cv2.dft(np.float32(encrypt1), flags=cv2.DFT_COMPLEX_OUTPUT)
encrypt_fft2 = cv2.dft(np.float32(encrypt2), flags=cv2.DFT_COMPLEX_OUTPUT)
encrypt_fft3 = cv2.dft(np.float32(encrypt3), flags=cv2.DFT_COMPLEX_OUTPUT)

# 装载第一张图片
carrier_fft[128: 256, 0: 128] = encrypt_fft1[0: 128, 0: 128] / 500
carrier_fft[256: 384, 0: 128] = encrypt_fft1[384: 512, 0: 128] / 500
carrier_fft[0: 128, 128: 256] = encrypt_fft1[0: 128, 384: 512] / 500
carrier_fft[128: 256, 128: 256] = encrypt_fft1[384: 512, 384: 512] / 500
# 装载第二张图片
carrier_fft[256: 384, 128: 256] = encrypt_fft2[0: 128, 0: 128] / 500
carrier_fft[384: 512, 128: 256] = encrypt_fft2[384: 512, 0: 128] / 500
carrier_fft[0: 128, 256: 384] = encrypt_fft2[0: 128, 384: 512] / 500
carrier_fft[128: 256, 256: 384] = encrypt_fft2[384: 512, 384: 512] / 500
# 装载第三张图片
carrier_fft[256: 384, 256: 384] = encrypt_fft3[0: 128, 0: 128] / 500
carrier_fft[384: 512, 256: 384] = encrypt_fft3[384: 512, 0: 128] / 500
carrier_fft[128: 256, 384: 512] = encrypt_fft3[0: 128, 384: 512] / 500
carrier_fft[256: 384, 384: 512] = encrypt_fft3[384: 512, 384: 512] / 500

carrier_fft_ifft = cv2.idft(carrier_fft)
carrier_fft_ifft = cv2.magnitude(carrier_fft_ifft[:, :, 0], carrier_fft_ifft[:, :, 1])
carrier_fft_ifft_img = cv2.normalize(carrier_fft_ifft, carrier_fft_ifft, 0, 1, cv2.NORM_MINMAX)
plt.subplot(221)
plt.imshow(carrier_fft_ifft_img, cmap='gray')

carrier_fft_ifft_img_fft = cv2.dft(np.float32(carrier_fft_ifft_img), flags=cv2.DFT_COMPLEX_OUTPUT)

a = np.zeros((512, 512, 2))
__encrypt_fft1 = a
__encrypt_fft2 = a
__encrypt_fft3 = a
__proto_image = a

# 拆解第一张图片
__encrypt_fft1[0: 128, 0: 128] = carrier_fft_ifft_img_fft[128: 256, 0: 128] * 500
__encrypt_fft1[384: 512, 0: 128] = carrier_fft_ifft_img_fft[256: 384, 0: 128] * 500
__encrypt_fft1[0: 128, 384: 512] = carrier_fft_ifft_img_fft[0: 128, 128: 256] * 500
__encrypt_fft1[384: 512, 384: 512] = carrier_fft_ifft_img_fft[128: 256, 128: 256] * 500
__encrypt_fft1_ifft = cv2.idft(__encrypt_fft1)
__encrypt_fft1_ifft = cv2.magnitude(__encrypt_fft1_ifft[:, :, 0], __encrypt_fft1_ifft[:, :, 1])
__encrypt_fft1_ifft_img = cv2.normalize(__encrypt_fft1_ifft, __encrypt_fft1_ifft, 0, 1, cv2.NORM_MINMAX)
plt.subplot(222)
plt.imshow(__encrypt_fft1_ifft_img, cmap='gray')

# # 拆解第二张图片
# __encrypt_fft2[0: 128, 0: 128] = carrier_fft2[256: 384, 128: 256] * 500
# __encrypt_fft2[384: 512, 0: 128] = carrier_fft2[384: 512, 128: 256] * 500
# __encrypt_fft2[0: 128, 384: 512] = carrier_fft2[0: 128, 256: 384] * 500
# __encrypt_fft2[384: 512, 384: 512] = carrier_fft2[128: 256, 256: 384] * 500
# # 拆解第三张图片
# __encrypt_fft3[0: 128, 0: 128] = carrier_fft2[256: 384, 256: 384] * 500
# __encrypt_fft3[384: 512, 0: 128] = carrier_fft2[384: 512, 256: 384] * 500
# __encrypt_fft3[0: 128, 384: 512] = carrier_fft2[128: 256, 384: 512] * 500
# __encrypt_fft3[384: 512, 384: 512] = carrier_fft2[256: 384, 384: 512] * 500

__proto_image[0: 128, 0: 128] = carrier_fft_ifft_img_fft[0: 128, 0: 128]
__proto_image[384: 512, 0: 128] = carrier_fft_ifft_img_fft[384: 512, 0: 128]
__proto_image[0: 128, 384: 512] = carrier_fft_ifft_img_fft[0: 128, 384: 512]
__proto_image[384: 512, 384: 512] = carrier_fft_ifft_img_fft[384: 512, 384: 512]
__proto_image_ifft = cv2.idft(__proto_image)
__proto_image_ifft = cv2.magnitude(__proto_image_ifft[:, :, 0], __proto_image_ifft[:, :, 1])
__proto_image_ifft_img = cv2.normalize(__proto_image_ifft, __proto_image_ifft, 0, 1, cv2.NORM_MINMAX)
plt.subplot(223)
plt.imshow(__proto_image_ifft_img, cmap='gray')

plt.show()
