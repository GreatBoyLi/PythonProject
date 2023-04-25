import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from skimage import io, exposure, filters


carrier = io.imread('../image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('../image/encrypt/encrypt1.pgm', as_gray=True)
encrypt2 = io.imread('../image/encrypt/encrypt2.pgm', as_gray=True)
encrypt3 = io.imread('../image/encrypt/encrypt3.pgm', as_gray=True)


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


carrier_ifft_img = ((np.abs(fft.ifft2(carrier_fft))))


a = fft.ifft2(carrier_fft)
print((a.imag).min(), (a.imag).max())
test = np.ones((512, 512, 3), dtype=np.uint8)
for x in range(a.shape[0]):
    for y in range(a.shape[1]):
        if int(a[x][y].real) > 255:
            test[x][y][0] = 255
        else:
            test[x][y][0] = int(a[x][y].real)
        if a[x][y].imag < 0:
            test[x][y][1] = (a[x][y].imag*-100).astype(np.uint8)
            test[x][y][2] = 2
        else:
            test[x][y][1] = (a[x][y].imag * 100).astype(np.uint8)
print(test[:,:,1].max(), test[:,:,1].min())

io.imsave('../image/liwp2.png', test)
b = io.imread('../image/liwp2.png')

c = np.zeros((512, 512))
c = c + c * 1j
for x in range(b.shape[0]):
    for y in range(b.shape[1]):
        if b[x][y][2] == 2:
            c[x][y] = b[x][y][0] -(b[x][y][1] / 100) * 1j
        else:
            c[x][y] = b[x][y][0] + (b[x][y][1] / 100) * 1j



carrier_fft2 = fft.fft2(c)


# 傅里叶变换后的加密图像
fftImg2 = 20 * np.log(np.abs(carrier_fft2))

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

axe1 = exposure.equalize_hist(axe1)
axe1 = filters.median(axe1)


axe[0].imshow(axe0, cmap='gray')
axe[1].imshow(axe1, cmap='gray')
axe[2].imshow(axe2, cmap='gray')
axe[3].imshow(axe3, cmap='gray')


plt.savefig("../image/liwp1.png")
plt.show()
