import numpy as np
import cv2
import testSIFT
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10

img1 = cv2.imread('../Image/1.jpg', 0)
img2 = cv2.imread('../Image/2.jpg', 0)

# Compute SIFT keypoints and descriptors
kp1, des1 = testSIFT.computeKeypointsAndDescriptors(img1)
kp2, des2 = testSIFT.computeKeypointsAndDescriptors(img2)

# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
"""
    返回的俩个DMatch数据结构类型，
        queryIdx，trainIdx，distance
        queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        distance：代表这一次匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。----相似性的度量
    这俩个DMatch数据类型是俩个与原图像特征点最接近的俩个特征点（match返回的是最匹配的）
    只有这俩个特征点的欧式距离小于一定值的时候才会认为匹配成功。
"""
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        # 匹配成功放入good
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # 组合两幅图
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    plt.imshow(newimg)
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

