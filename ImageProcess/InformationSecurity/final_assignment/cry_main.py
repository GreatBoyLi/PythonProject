import rsa
import Photo
from matplotlib import pyplot as plt
from skimage import io

if __name__ == "__main__":
    # 读取图片
    path = 'C:/Users/Great_Boy_Li/Desktop/portray.jpg'
    oriImage = io.imread(path)
    rad1 = Photo.Random(oriImage)

    # 打乱图片
    image1 = rad1.shuffle(oriImage)
    plt.subplot(121)
    plt.imshow(image1, cmap='gray')
    print("1")
    # 生成密钥
    pubkey, prikey = rsa.newkeys(3300)
    print("2")
    # 加密
    en_array = rad1.ency_array(pubkey)
    print("3")
    # 恢复图片
    rad2 = Photo.Random(oriImage)
    rad2.array = rad2.decy_array(prikey, en_array)
    print(rad2.array)
    print("4")
    image2 = rad2.recovery(image1)
    print("5")
    plt.subplot(122)
    plt.imshow(image2, cmap='gray')

    plt.show()

