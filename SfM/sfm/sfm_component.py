import cv2
import numpy as np


def extract_features(image_names: list):
    """
    提取两两图片中的sift特征，返回对应的特征点，描述符和颜色
    :param image_names:
    :return: (keypoints, descriptors, colors)
    """
    # 全部使用默认值
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_all = []
    descriptor_all = []
    color_all = []
    for image_name in image_names:
        image = cv2.imread(image_name)
        if image is None:
            continue
        # 提取sift特征
        keypoints, descriptor = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        color = np.zeros((len(keypoints), 3))
        if len(keypoints) < 10:
            continue
        keypoints_all.append(keypoints)
        descriptor_all.append(descriptor)
        for i, keypoint in enumerate(keypoints):
            p = keypoint.pt
            color[i] = image[int(p[1])][int(p[0])]
        color_all.append(color)
    return np.array(keypoints_all, object), np.array(descriptor_all, object), np.array(color_all, object)
