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


def getMatchedPoints(keyPoint1, keyPoint2, match):
    """
    获得匹配的SIFT特征点的x，y坐标
    :param keyPoint1: 第一个SIFT特征的keyPoint列表
    :param keyPoint2: 第二个SIFT特征的keyPoint列表
    :param match: 两张图片SIFT特征匹配的数据
    :return: 返回两张图片匹配的SIFT特征点的坐标
    """
    srcPts = []
    dstPts = []
    for m in match:
        srcPts.append(keyPoint1[m.queryIdx].pt)
        dstPts.append(keyPoint2[m.trainIdx].pt)
    return np.array(srcPts), np.array(dstPts)


def getMatchedColors(colorPoint1, colorPoint2, match):
    """
    获得匹配的SIFT点的color值
    :param colorPoint1:
    :param colorPoint2:
    :param match:
    :return:
    """
    srcColor = []
    dstColor = []
    for m in match:
        srcColor.append(colorPoint1[m.queryIdx])
        dstColor.append(colorPoint2[m.trainIdx])
    return np.array(srcColor), np.array(dstColor)


def init_structure(K, keypoints_all, color_all, matches_all):
    """

    :param K: 相机内参矩阵
    :param keypoints_all: SIFT检测出的特征点
    :param color_all: 特征点对应的颜色
    :param matches_all: 匹配的特征点
    :return:
    """
    # 两张图片的特征点坐标
    p1, p2 = getMatchedPoints(keypoints_all[0], keypoints_all[1], matches_all[0])
    c1, c2 = getMatchedColors(color_all[0], color_all[1], matches_all[0])

    # 寻找两张图片之间对应相机旋转角度以及相机平移
    focal_length = K[0][0]
    principle_point = (K[0][2], K[1][2])
    # 两种计算本质矩阵的方法都可以，计算的结果一样
    E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    # E1, mask1 = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0, mask=None)
    _, R, T, mask = cv2.recoverPose(E, p1, p2, K, mask)
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


def fusionStructure(matches, structIndices, nextStructIndices, structure, nextStruct, colors, nextColors):
    """
    将已经作出的点云进行融合
    :param matches: SIFT特征匹配的点
    :param structIndices: 已经三角化的点以及顺序（大于0的下标是三角化的点，大于0的值也代表着点云的顺序）
    :param nextStructIndices: 下一张图片三角化的点以及顺序
    :param structure: 三角化点的集合
    :param nextStruct: 下一张图片三角化点的集合
    :param colors:
    :param nextColors:
    :return:
    """
    for i, match in enumerate(matches):
        queryIdx = match.queryIdx
        trainIdx = match.trainIdx
        structIdx = structIndices[queryIdx]
        if structIdx > 0:
            nextStructIndices[trainIdx] = structIdx
            continue
        structure = np.append(structure, [nextStruct[i]], axis=0)
        colors = np.append(colors, [nextColors[i]], axis=0)
        structIndices[queryIdx] = nextStructIndices[trainIdx] = len(structure) - 1
    return structIndices, nextStructIndices, structure, colors

