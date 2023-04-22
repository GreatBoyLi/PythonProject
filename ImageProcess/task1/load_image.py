from numpy import fft
import numpy as np
from skimage import io
from matplotlib import pyplot as plt


class loadImage:
    __carrier_image = None
    __encrypt_image1 = None
    __encrypt_image2 = None
    __encrypt_image3 = None

    def __init__(self, carrier_image, encrypt_image1, encrypt_image2, encrypt_image3):
        self.__carrier_image = carrier_image
        self.__encrypt_image1 = encrypt_image1
        self.__encrypt_image2 = encrypt_image2
        self.__encrypt_image3 = encrypt_image3

    def load_image(self, is_encrypt=False):
        if is_encrypt:
            pass
        else:
            pass
        # 傅里叶变换
        carrier_fft = fft.fft2(self.__carrier_image)
        encrypt_fft1 = fft.fft2(self.__encrypt_image1)
        encrypt_fft2 = fft.fft2(self.__encrypt_image2)
        encrypt_fft3 = fft.fft2(self.__encrypt_image3)

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

        # 逆傅里叶变换
        return np.abs(fft.ifft2(carrier_fft))


class unLoadImage:
    __carrier_image = None
    __proto_image = None
    __encrypt_fft1 = None
    __encrypt_fft2 = None
    __encrypt_fft3 = None

    def __init__(self, carrier_image):
        self.__carrier_image = carrier_image
        a = np.zeros((512, 512))
        self.__encrypt_fft1 = a + a * 1j
        self.__encrypt_fft2 = a + a * 1j
        self.__encrypt_fft3 = a + a * 1j
        self.__proto_image = a + a * 1j

    def unload_image(self):
        carrier_fft = fft.fft2(self.__carrier_image)

        # 拆解第一张图片
        self.__encrypt_fft1[0: 128, 0: 128] = carrier_fft[128: 256, 0: 128] * 500
        self.__encrypt_fft1[384: 512, 0: 128] = carrier_fft[256: 384, 0: 128] * 500
        self.__encrypt_fft1[0: 128, 384: 512] = carrier_fft[0: 128, 128: 256] * 500
        self.__encrypt_fft1[384: 512, 384: 512] = carrier_fft[128: 256, 128: 256] * 500
        # 拆解第二张图片
        self.__encrypt_fft2[0: 128, 0: 128] = carrier_fft[256: 384, 128: 256] * 500
        self.__encrypt_fft2[384: 512, 0: 128] = carrier_fft[384: 512, 128: 256] * 500
        self.__encrypt_fft2[0: 128, 384: 512] = carrier_fft[0: 128, 256: 384] * 500
        self.__encrypt_fft2[384: 512, 384: 512] = carrier_fft[128: 256, 256: 384] * 500
        # 拆解第三张图片
        self.__encrypt_fft3[0: 128, 0: 128] = carrier_fft[256: 384, 256: 384] * 500
        self.__encrypt_fft3[384: 512, 0: 128] = carrier_fft[384: 512, 256: 384] * 500
        self.__encrypt_fft3[0: 128, 384: 512] = carrier_fft[128: 256, 384: 512] * 500
        self.__encrypt_fft3[384: 512, 384: 512] = carrier_fft[256: 384, 384: 512] * 500

        # 拆解载图图片
        self.__proto_image[0: 128, 0: 128] = carrier_fft[0: 128, 0: 128]
        self.__proto_image[384: 512, 0: 128] = carrier_fft[384: 512, 0: 128]
        self.__proto_image[0: 128, 384: 512] = carrier_fft[0: 128, 384: 512]
        self.__proto_image[384: 512, 384: 512] = carrier_fft[384: 512, 384: 512]

        return np.abs(fft.ifft2(self.__proto_image)), np.abs(fft.ifft2(self.__encrypt_fft1)), np.abs(
            fft.ifft2(self.__encrypt_fft2)), np.abs(fft.ifft2(self.__encrypt_fft3))


if __name__ == '__main__':
    carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
    encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)
    encrypt2 = io.imread('./image/encrypt/encrypt2.pgm', as_gray=True)
    encrypt3 = io.imread('./image/encrypt/encrypt3.pgm', as_gray=True)

    loadImg = loadImage(carrier, encrypt1, encrypt2, encrypt3)
    encryptedImg = loadImg.load_image()

    fig, axes = plt.subplots(2, 2)
    axe = axes.ravel()

    unloadImg = unLoadImage(encryptedImg)
    protoImg, unencryptImg1, unencryptImg12, unencryptImg3 = unloadImg.unload_image()

    axe[0].imshow(protoImg, cmap='gray')
    axe[1].imshow(unencryptImg1, cmap='gray')
    axe[2].imshow(unencryptImg12, cmap='gray')
    axe[3].imshow(unencryptImg3, cmap='gray')

    plt.savefig('./image/save_load_image.png')
    plt.show()
