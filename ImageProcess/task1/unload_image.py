from numpy import fft
import numpy as np


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

        return np.abs(fft.ifft2(self.__proto_image)), np.abs(fft.ifft2(self.__encrypt_fft1)), np.abs(fft.ifft2(self.__encrypt_fft2)), np.abs(fft.ifft2(self.__encrypt_fft3))
