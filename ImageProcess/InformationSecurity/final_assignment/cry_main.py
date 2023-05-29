import rsa
import Photo
from matplotlib import pyplot as plt
from skimage import io
from matplotlib.font_manager import FontProperties
import time

if __name__ == "__main__":
    # 读取图片
    chinese = FontProperties(fname='../../font/SourceHanSansSC-Normal.otf', size=13)
    path = 'C:/Users/Great_Boy_Li/Desktop/portray.jpg'
    oriImage = io.imread(path)
    rad1 = Photo.Random(oriImage)

    # plt.subplot(121)
    # plt.imshow(oriImage, cmap='gray')
    # plt.title("未打乱的图像", fontproperties=chinese)

    # 生成密钥
    pubkey, prikey = rsa.newkeys(3300)

    # 打乱图片
    start = time.time()
    image1 = rad1.shuffle(oriImage)
    plt.subplot(121)
    plt.figtext(0.5, 0.05, "图2 python自带的shuffle打乱", ha='center', fontsize=12, fontweight='bold', fontproperties=chinese)
    plt.title("打乱后的图像", fontproperties=chinese)

    plt.imshow(image1, cmap='gray')

    # 加密
    en_array = rad1.ency_array(pubkey)
    end = time.time()
    dura = end - start
    print(f"加密时间为：{dura}")

    start = time.time()
    # 恢复图片
    rad2 = Photo.Random(oriImage)
    rad2.array = rad2.decy_array(prikey, en_array)
    image2 = rad2.recovery(image1)
    end = time.time()
    dura = end - start
    print(f"解密时间为：{dura}")
    plt.subplot(122)
    plt.imshow(image2, cmap='gray')
    # plt.savefig("./images/2.png", dpi=500)
    plt.show()

