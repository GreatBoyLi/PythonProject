import config
import os
import sfm_component


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

sfm_component.init_structure(K, keypoints_all, color_all, matches_all)

print("111111")
