import exifread
import numpy as np
import pandas as pd

def get_camera_parameters(image_path, dict):
    f = open(image_path, 'rb')
    tags = exifread.process_file(f)
    f.close()

    # 获取 Exif 信息中的相机内部参数
    focal_length = float(tags['EXIF FocalLength'].values[0].num) / tags['EXIF FocalLength'].values[0].den
    pixel_width = float(tags['EXIF ExifImageWidth'].values[0])
    pixel_height = float(tags['EXIF ExifImageLength'].values[0])

    company = str(tags['Image Make']) + ' ' + str(tags['Image Model'])
    if company in dict:
        ccdw = float(dict[company])

    # 计算相机的内部参数矩阵
    fx = fy = focal_length * max(pixel_width, pixel_height) / ccdw
    cx = pixel_width / 2
    cy = pixel_height / 2

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return camera_matrix


def get_dict():
    # 读取文件filename
    df = pd.read_csv(filename)
    # 循环遍历文件中的每一行内容
    # print(df.iloc[:, [0]])
    a = df.iloc[:, [0]].values
    com_dict = {}
    for x in a:
        s = str(x[0])
        b = s.split(';')
        com_dict[b[0]] = b[1]
    return com_dict


filename = 'E:/3D_Reconstruction/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt'
path = 'Image/SceauxCastle/100_7100.JPG'

com_dict = get_dict()
a = get_camera_parameters(path, com_dict)
print(a)





