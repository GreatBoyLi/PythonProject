import random
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import liwp_cryptography as liwp


class Random:
    def __init__(self, image, array=None):
        self.count = image.shape
        if len(self.count) == 2:
            self.height, self.width = image.shape
            self.size = self.width * self.height
            if array is not None:
                self.array = array
            else:
                self.array = [i for i in range(self.size)]
            random.shuffle(self.array)
        else:
            self.height, self.width, self.channel = image.shape
            self.size = self.width * self.height * self.channel
            if array is not None:
                self.array = array
            else:
                self.array = [i for i in range(self.size)]
            random.shuffle(self.array)

    def shuffle(self, image):
        flat_image = image.reshape(self.size)
        new_image = np.zeros_like(flat_image)
        for i in range(self.size):
            new_image[i] = flat_image[self.array[i]]
        if len(self.count) == 3:
            return new_image.reshape(self.height, self.width, self.channel)
        return new_image.reshape(self.height, self.width)

    def recovery(self, image):
        flat_image = image.reshape(self.size)
        new_image = np.zeros_like(flat_image)
        for i in range(self.size):
            new_image[self.array[i]] = flat_image[i]
        if len(self.count) == 3:
            return new_image.reshape(self.height, self.width, self.channel)
        return new_image.reshape(self.height, self.width)

    def ency_array(self, public_key):
        enc_array = []
        last = 0
        rs_obj = liwp.RsaCrypt(public_key)
        for i in range(0, self.size - 50, 50):
            array = self.array[i:i+50]
            last = i + 50
            text = str(array)
            ency_text = rs_obj.encrypt(text)
            enc_array.append(ency_text)
        array = self.array[last:]
        text = str(array)
        ency_text = rs_obj.encrypt(text)
        enc_array.append(ency_text)
        return enc_array

    def decy_array(self, private_key, enc_array):
        dec_array = []
        rs_obj = liwp.RsaCrypt(None, private_key)
        for ency_text in enc_array:
            decy_text = rs_obj.decrypt(ency_text)
            decy_text = decy_text.replace(" ", "").replace("[", "").replace("]", "")
            array = decy_text.split(",")
            dec_array.extend([int(i) for i in array])
        return dec_array


if __name__ == "__main__":
    path = 'C:/Users/Great_Boy_Li/Desktop/portray.jpg'
    oriImage = io.imread(path, as_gray=True)

    rad = Random(oriImage)
    print(len(rad.array))
    image1 = rad.shuffle(oriImage)

    image2 = rad.recovery(image1)

    plt.subplot(121)
    plt.imshow(image1, cmap='gray')
    plt.subplot(122)
    plt.imshow(image2, cmap='gray')

    plt.show()

