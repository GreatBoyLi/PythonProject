import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from skimage import io


carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)
encrypt2 = io.imread('./image/encrypt/encrypt2.pgm', as_gray=True)
encrypt3 = io.imread('./image/encrypt/encrypt3.pgm', as_gray=True)


# 傅里叶变换
carrier_fft = fft.fft2(carrier)
encrypt_fft1 = fft.fft2(encrypt1)
encrypt_fft2 = fft.fft2(encrypt2)
encrypt_fft3 = fft.fft2(encrypt3)

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

# 加密过后的载体图像傅里叶频谱
fftImg1 = 20 * np.log(np.abs(carrier_fft))

carrier_ifft_img = ((np.abs(fft.ifft2(carrier_fft))))
test1 = np.ceil((carrier_ifft_img - np.min(carrier_ifft_img)) /
                (np.max(carrier_ifft_img) - np.min(carrier_ifft_img)) * 255)
test2 = np.ceil((np.abs(fft.ifft2(carrier_fft)))) * 255 * 255
angle = np.floor(np.angle(fft.ifft2(carrier_fft)) * 100000)
# carrier_ifft_img = fft.ifft2(carrier_fft)
# m = carrier_ifft_img * np.cos(angle) + carrier_ifft_img * np.sin(angle) * 1j
m = carrier_ifft_img * np.cos(angle / 100000) + carrier_ifft_img * np.sin(angle / 100000) * 1j
# carrier_fft2 = fft.fft2(test1)
a = fft.ifft2(carrier_fft)
d = np.abs(a)
carrier_fft2 = fft.fft2(a)

anylize = carrier_fft / carrier_fft2

# 傅里叶变换后的加密图像
fftImg2 = 20 * np.log(np.abs(carrier_fft2))
# fftImg4 = 20 * np.log(np.abs(fft.fftshift(carrier_fft2)))

count = 0
for x in range(512):
    for y in range(512):
        if carrier_fft[x][y] != carrier_fft2[x][y]:
            count += 1

a = np.zeros((512, 512))
__encrypt_fft1 = a + a * 1j
__encrypt_fft2 = a + a * 1j
__encrypt_fft3 = a + a * 1j
__proto_image = a + a * 1j

# 拆解第一张图片
__encrypt_fft1[0: 128, 0: 128] = carrier_fft2[128: 256, 0: 128] * 500
__encrypt_fft1[384: 512, 0: 128] = carrier_fft2[256: 384, 0: 128] * 500
__encrypt_fft1[0: 128, 384: 512] = carrier_fft2[0: 128, 128: 256] * 500
__encrypt_fft1[384: 512, 384: 512] = carrier_fft2[128: 256, 128: 256] * 500
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


fig, axes = plt.subplots(2, 2)
axe = axes.ravel()

a0 = fft.ifft2(__proto_image)
a1 = fft.ifft2(__encrypt_fft1)
a2 = fft.ifft2(__encrypt_fft2)
a3 = fft.ifft2(__encrypt_fft3)

axe0 = np.ceil(np.abs(a0)).astype(int)
axe1 = np.ceil(np.abs(a1)).astype(int)
axe2 = np.ceil(np.abs(a2)).astype(int)
axe3 = np.ceil(np.abs(a3)).astype(int)


# axe[0].imshow(test2, cmap='gray')
# axe[1].imshow(axe0, cmap='gray')
# axe[2].imshow(axe1, cmap='gray')
# axe[3].imshow(axe2, cmap='gray')
# axe[4].imshow(axe3, cmap='gray')
# axe[5].imshow(fftImg1)
# axe[6].imshow(fftImg2)

# axe[0].imshow(test2, cmap='gray')
axe[0].imshow(axe0, cmap='gray')
axe[1].imshow(axe1, cmap='gray')
axe[2].imshow(axe2, cmap='gray')
axe[3].imshow(axe3, cmap='gray')
# axe[5].imshow(fftImg1)
# axe[6].imshow(fftImg2)

# axe[7].imshow(fftImg4)

# io.imsave('./image/test.jpg', carrier_ifft_img)
test = io.imread('./image/test.jpg', as_gray=True)
plt.savefig("./image/save_test8.png")
plt.show()
