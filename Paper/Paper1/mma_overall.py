import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LoFTR_y = \
    [0, 0.50, 0.73, 0.82, 0.85, 0.86, 0.877, 0.888, 0.895, 0.903, 0.91]
MatchFormer_y = \
    [0, 0.45, 0.69, 0.78, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]

SwinMatcher_y = \
    [0, 0.53, 0.75, 0.84, 0.86, 0.879, 0.889, 0.899, 0.908, 0.915, 0.92]

sift_superglue_y = \
    [0, 0.30, 0.49, 0.61, 0.67, 0.71, 0.72, 0.74, 0.75, 0.76, 0.77]

orb_gms_y = \
    [0, 0.28, 0.46, 0.57, 0.63, 0.66, 0.68, 0.69, 0.70, 0.72, 0.73]

sp_pointcn_y = \
    [0, 0.32, 0.55, 0.679, 0.75, 0.799, 0.810, 0.82, 0.83, 0.845, 0.852]
sp_superglue_y = \
    [0, 0.50, 0.72, 0.81, 0.86, 0.88, 0.90, 0.91, 0.925, 0.935, 0.94]

plt.subplots(figsize=(6.5, 10))
linewidth = 1.5

# 创建曲线图
plt.plot(x, LoFTR_y, color='cyan', label='LoFTR', linewidth=linewidth)
plt.plot(x, np.array(MatchFormer_y), color='red', label='MatchFormer', linewidth=linewidth)
plt.plot(x, np.array(SwinMatcher_y), color='green', label='SwinMatcher', linewidth=linewidth)
plt.plot(x, np.array(sift_superglue_y), color='blue', label='SIFT+SuperGlue', linewidth=linewidth)
plt.plot(x, np.array(orb_gms_y), color='orange', label='ORB+GMS', linewidth=linewidth)
plt.plot(x, np.array(sp_pointcn_y), color='purple', label='SuperPoint+PointCN', linewidth=linewidth)
plt.plot(x, np.array(sp_superglue_y), color='black', label='SuperPoint+SuperGlue', linewidth=linewidth)

plt.xticks([0,1,2, 3,4, 5, 6,7,8, 9,10])
plt.yticks([0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9, 1.0])
plt.grid(True)
plt.xlim(1,10)  # X轴的界限从0开始
plt.ylim(0.3,1.0)

ax = plt.gca()
# 不显示上面和右边的边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.tick_params(labelsize=20)

# 添加标题
plt.title("Overall",fontdict={'family': 'Times New Roman', 'size': 25})

# 添加X轴和Y轴标签
plt.xlabel("Thresholds(px)",fontdict={'family': 'Times New Roman', 'size': 25})
plt.ylabel("MMA",fontdict={'family': 'Times New Roman', 'size': 25})


plt.legend(loc='best', framealpha=0,prop={'family': 'Times New Roman', 'size': 18})

plt.savefig('C:\\Users\\leewe\\Desktop\\工作\\论文图像\\5.9.png', dpi=600)
# 显示图表
plt.show()
