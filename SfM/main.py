# import cv2
#
# img = cv2.imread("C:/Users/Great_Boy_Li/Desktop/123.jpg")
#
# resize_img = cv2.resize(img, (0, 0), fx=4, fy=4)
#
# cv2.imwrite("C:/Users/Great_Boy_Li/Desktop/1234.jpg", resize_img)


# test from windows
# test from Mac

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

path = "/Users/lee/Desktop"

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
chinese = FontProperties(fname='../font/SourceHanSansSC-Normal.otf', size=13)

x = np.linspace(0, 7, 200)
y = np.sin(x)
y1 = np.cos(x)
y2 = x**2
y3 = x**-1

plt.subplot(221)
plt.plot(x, y, label="y=sin(x)", color="red", linewidth=2)
plt.legend(title="test", prop=chinese, loc="upper right")
plt.subplot(222)
plt.plot(x, y1, label="$y=cos(x)$", color="blue", linewidth=2)
plt.legend(title="test", prop=chinese, loc="upper right")
plt.subplot(223)
plt.plot(x, y2, label=r"$y=x^2$", color="green", linewidth=2)
plt.legend(title="test", prop=chinese, loc="upper right")
plt.subplot(224)
plt.plot(x, y3, label=r"$y=x^{-1}$", color="black", linewidth=2)
plt.legend(title="test", prop=chinese, loc="upper right")
plt.title("正弦函数", fontproperties=chinese)
# plt.legend(title="test", prop=chinese, loc="upper right")
plt.xlabel("x轴", fontproperties=chinese)
plt.ylabel("y轴", fontproperties=chinese)
plt.ylim(-1, 2)
ax = plt.gca()
ax.spines['right'].set_color('blue')
plt.savefig(path + "/test.png")
plt.show()
