import numpy as np
from numpy import fft
from skimage import io
from matplotlib import pyplot as plt
import cv2


carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)

carrier_fft = cv2.dft(np.float32(carrier), flags=cv2.DFT_COMPLEX_OUTPUT)
encrypt_fft1 = cv2.dft(np.float32(encrypt1), flags=cv2.DFT_COMPLEX_OUTPUT)


# 装载第一张图片
carrier_fft[169: 341, 0: 256] = encrypt_fft1[0: 172, 0: 256] / 500
carrier_fft[0: 169, 169: 341] = encrypt_fft1[343: 512, 0: 172] / 500
carrier_fft[169: 341, 256: 512] = encrypt_fft1[0: 172, 256: 512] / 500
carrier_fft[341: 512, 169: 341] = encrypt_fft1[341: 512, 340: 512] / 500

carrier_fft_ifft = cv2.idft(carrier_fft)
carrier_fft_ifft = cv2.magnitude(carrier_fft_ifft[:, :, 0], carrier_fft_ifft[:, :, 1])
d = cv2.normalize(carrier_fft_ifft, carrier_fft_ifft, 0, 1, cv2.NORM_MINMAX)

carrier_fft_ifft_fft = cv2.dft(np.float32(carrier_fft_ifft), flags=cv2.DFT_COMPLEX_OUTPUT)

a = np.zeros((512, 512, 2))
__encrypt_fft1 = a
__proto_image = a

# 拆解第一张图片
__encrypt_fft1[0: 172, 0: 256] = carrier_fft_ifft_fft[169: 341, 0: 256] * 500
__encrypt_fft1[343: 512, 0: 172] = carrier_fft_ifft_fft[0: 169, 169: 341] * 500
__encrypt_fft1[0: 172, 256: 512] = carrier_fft_ifft_fft[169: 341, 256: 512] * 500
__encrypt_fft1[341: 512, 340: 512] = carrier_fft_ifft_fft[341: 512, 169: 341] * 500

__proto_image[0: 169, 0: 169] = carrier_fft_ifft_fft[0: 169, 0: 169]
__proto_image[341: 512, 0: 169] = carrier_fft_ifft_fft[341: 512, 0: 169]
__proto_image[0: 169, 341: 512] = carrier_fft_ifft_fft[0: 169, 341: 512]
__proto_image[341: 512, 341: 512] = carrier_fft_ifft_fft[341: 512, 341: 512]

plt.subplot(221)
plt.imshow(d, cmap='gray')


__proto_image_ifft = cv2.idft(__proto_image)
__proto_image_ifft = cv2.magnitude(__proto_image_ifft[:, :, 0], __proto_image_ifft[:, :, 1])
t = cv2.normalize(__proto_image_ifft, __proto_image_ifft, 0, 1, cv2.NORM_MINMAX)

plt.subplot(222)
plt.imshow(t, cmap='gray')

plt.show()
