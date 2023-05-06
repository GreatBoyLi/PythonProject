import numpy as np

import config
import os
import sfm_component
import cv2


image_dir = config.image_dir
image_names = os.listdir(image_dir)
image_names = sorted(image_names)
for i in range(len(image_names)):
    image_names[i] = image_dir + '/' + image_names[i]

# 获得相机内参数矩阵
K = config.getK()

# 获取图片两两对应的sift特征keypoints、descriptor和坐标对应的三通道color
keypoints_all, descriptor_all, color_all = sfm_component.extract_features(image_names)
# 获取图片两两对应的特征点匹配
matches_all = sfm_component.match_all_features(descriptor_all)

structure, correspond_structIdx, colors, rotations, motions = \
    sfm_component.init_structure(K, keypoints_all, color_all, matches_all)

# 两两图片循环获得对应点的信息
for i in range(1, len(matches_all)):
    object_points, image_points = \
        sfm_component.get_objpoints_and_imgpoints(matches_all[i], correspond_structIdx[i], structure, keypoints_all[i+1])
    # 在python的opencv中solvePnpRansac函数的第一个码数长度要大于7，否则会报错
    # 这里对小于7的点集做一个重复填充操作，即用点集中的第一个点补满7个
    while len(image_points) < 7:
        object_points = np.append(object_points, [object_points[0]], axis=0)
        image_points = np.append(image_points, [image_points[0]], axis=0)

    _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))
    R, _ = cv2.Rodrigues(r)
    rotations.append(R)
    motions.append(T)
    # 获得图片匹配特征点的keyPoint数据
    print(i)
    p1, p2 = sfm_component.getMatchedPoints(keypoints_all[i], keypoints_all[i+1], matches_all[i])
    c1, c2 = sfm_component.getMatchedColors(color_all[i], color_all[i+1], matches_all[i])
    # 三维重建下一对图片的
    nextStructure = sfm_component.reconstruct(K, rotations[i], motions[i], R, T, p1, p2)
    correspond_structIdx[i], correspond_structIdx[i+1], structure, colors = \
        sfm_component.fusionStructure(matches_all[i], correspond_structIdx[i], correspond_structIdx[i+1],
                                      structure, nextStructure, colors, c1)

structure = sfm_component.bundle_adjustment(rotations, motions, K, correspond_structIdx, keypoints_all, structure)

print("111111")
