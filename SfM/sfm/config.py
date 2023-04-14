import numpy as np

image_dir = '../Image/SceauxCastle'


def getMRT():
    return 0.7


def getK():
    # 相机内参矩阵,其中，K[0][0]和K[1][1]代表相机焦距
    # 而K[0][2]和K[1][2]代表图像的中心像素。
    K = np.array([
            [2905.88, 0, 1416],
            [0, 2905.88,  1064],
            [0, 0, 1]])
    return K


# 选择性删除所选点的范围。
x = 0.5
y = 1
