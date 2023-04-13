import config
import os
import sfm_component


image_dir = config.image_dir
image_names = os.listdir(image_dir)
for i in range(len(image_names)):
    image_names[i] = image_dir + '/' + image_names[i]

# 获得相机内参数矩阵
K = config.getK()

# 获取图片两两对应的sift特征keypoints、descriptor和坐标对应的三通道color
keypoints_all, descriptor_all, color_all = sfm_component.extract_features(image_names)

print("111111")
