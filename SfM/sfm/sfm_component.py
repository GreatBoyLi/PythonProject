import cv2
import numpy as np
import config


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


def match_all_features(descriptor_all: np):
    """
    匹配所有的特征点
    :param descriptor_all: 特征点
    :return:
    """
    match_all = []
    for i in range(len(descriptor_all) - 1):
        matches = match_features(descriptor_all[i], descriptor_all[i + 1])
        match_all.append(matches)
    return np.array(match_all, object)


def match_features(query, train):
    """
    匹配两张图片的特征点
    :param query: 第一张图片
    :param train: 第二张图片
    :return:
    """
    # 最近邻匹配，取最近的两个
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(query, train, k=2)
    matches = []
    for m, n in knn_matches:
        # 低于阈值的匹配点，两个点的距离差距越大，匹配越好
        if m.distance < config.getMRT() * n.distance:
            matches.append(m)
    return matches


def init_structure(K, keypoints_all, color_all, matches_all):
    """

    :param K: 相机内参矩阵
    :param keypoints_all: SIFT检测出的特征点
    :param color_all: 特征点对应的颜色
    :param matches_all: 匹配的特征点
    :return:
    """
    # 两张图片的特征点坐标
    p1 = np.array([keypoints_all[0][m.queryIdx].pt for m in matches_all[0]])
    p2 = np.array([keypoints_all[1][m.trainIdx].pt for m in matches_all[0]])

    # 寻找两张图片之间对应相机旋转角度以及相机平移
    focal_length = K[0][0]
    principle_point = (K[0][2], K[1][2])
    # 两种计算本质矩阵的方法都可以，计算的结果一样
    E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    # E1, mask1 = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0, mask=None)
    _, R, t, mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    # 获得的mask不一样，数量也不一样，15号需要好好看看
    # 选择重合的点
    p3 = [p1[i] for i in range(len(mask)) if mask[i] > 0]
    p4 = [p2[i] for i in range(len(mask)) if mask[i] > 0]
    return p1, p2, R, t


# GitHub Copilot 的代码，保留
def get_essential_matrix(F, K):
    """
    计算本质矩阵
    :param F: 基础矩阵
    :param K: 相机内参矩阵
    :return: 本质矩阵
    """
    E = K.T @ F @ K
    U, S, V = np.linalg.svd(E)
    S[0] = 1
    S[1] = 1
    S[2] = 0
    E = U @ np.diag(S) @ V
    return E
