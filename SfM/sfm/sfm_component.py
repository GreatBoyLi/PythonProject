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
    c1 = np.array([color_all[0][m.queryIdx] for m in matches_all[0]])
    c2 = np.array([color_all[1][m.trainIdx] for m in matches_all[0]])

    # 寻找两张图片之间对应相机旋转角度以及相机平移
    focal_length = K[0][0]
    principle_point = (K[0][2], K[1][2])
    # 两种计算本质矩阵的方法都可以，计算的结果一样
    E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    # E1, mask1 = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0, mask=None)
    _, R, T, mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    # 获得的mask不一样，数量也不一样，15号需要好好看看
    # 选择重合的点
    p1 = np.array([p1[i] for i in range(len(mask)) if mask[i] > 0])
    p2 = np.array([p2[i] for i in range(len(mask)) if mask[i] > 0])
    colors = [c1[i] for i in range(len(mask)) if mask[i] > 0]
    # 设置第一个相机的变换矩阵，即作为剩下相机矩阵变换的基准
    R0 = np.eye(3)
    T0 = np.zeros((3, 1))
    # 三角化获得三维坐标
    struct = reconstruct(K, R0, T0, R, T, p1, p2)
    # 旋转和平移的矩阵
    rotations = [R0, R]
    motions = [T0, T]
    # 用到的点按顺序记录下来
    correspond_structIdx = []
    for kye_p in keypoints_all:
        correspond_structIdx.append(np.ones(len(kye_p)) * -1)
    inx = 0
    for i, match in enumerate(matches_all[0]):
        if mask[i] > 0:
            correspond_structIdx[0][int(match.queryIdx)] = inx
            correspond_structIdx[1][int(match.trainIdx)] = inx
            inx += 1
    return struct, correspond_structIdx, colors, rotations, motions


def reconstruct(K, R1, T1, R2, T2, p1, p2):
    """
    三角化重建
    :param K: 相机内参数
    :param R1: 第一个相机的旋转矩阵
    :param T1: 第一个相机的平移矩阵
    :param R2: 第二个相机的旋转矩阵
    :param T2: 第二个相机的平移矩阵
    :param p1: 第一张图片匹配的特征点
    :param p2: 第二张图片匹配的特征点
    :return: 点的三维坐标
    """
    proj1 = np.hstack((R1, T1), dtype=np.float32)
    proj2 = np.hstack((R2, T2), dtype=np.float32)
    proj1 = K.dot(proj1)
    proj2 = K.dot(proj2)
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    structure = []
    for i in range(s.shape[1]):
        structure.append((s[:, i] / s[3, i])[0:3])
    return np.array(structure, dtype=np.float32)


def get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):
    """
    制作图像点以及空间点
    :param matches:
    :param struct_indices:
    :param structure:
    :param key_points:
    :return:
    """
    object_points =[]
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx > 0:
            object_points.append(structure[int(struct_idx)])
            image_points.append(key_points[train_idx].pt)
    return np.array(object_points), np.array(image_points)
