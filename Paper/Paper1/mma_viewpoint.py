import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LoFTR_y = \
    [0, 0.415, 0.691, 0.795, 0.840, 0.889, 0.900, 0.902, 0.906, 0.910, 0.915]
MatchFormer_y = \
    [0, 0.389, 0.669, 0.78, 0.832, 0.889, 0.900, 0.902, 0.906, 0.910, 0.915]

SwinMatcher_y = \
    [0, 0.425, 0.700, 0.815, 0.854, 0.890, 0.909, 0.910, 0.909, 0.915, 0.920]

sift_superglue_y = \
    [0, 0.21, 0.49, 0.61, 0.69, 0.735, 0.760, 0.765, 0.770, 0.775, 0.780]

orb_gms_y = \
    [0, 0.19, 0.475, 0.589, 0.631, 0.679, 0.687, 0.690, 0.698, 0.709, 0.715]

sp_pointcn_y = \
    [0, 0.20, 0.57, 0.715, 0.80, 0.835, 0.850, 0.855, 0.859, 0.864, 0.870]
sp_superglue_y = \
    [0, 0.405, 0.760, 0.875, 0.924, 0.935, 0.940, 0.945, 0.955, 0.960, 0.965]

plt.subplots(figsize=(6.5, 10))
linewidth = 1.5

# 创建曲线图
plt.plot(x, LoFTR_y, color='cyan', label='LoFTR',linewidth=linewidth)
plt.plot(x, np.array(MatchFormer_y), color='red', label='MatchFormer',linewidth=linewidth)
plt.plot(x, np.array(SwinMatcher_y), color='green', label='SwinMatcher',linewidth=linewidth)
plt.plot(x, np.array(sift_superglue_y), color='blue', label='SIFT+SuperGlue',linewidth=linewidth)
plt.plot(x, np.array(orb_gms_y), color='orange', label='ORB+GMS',linewidth=linewidth)
plt.plot(x, np.array(sp_pointcn_y), color='purple', label='SuperPoint+PointCN',linewidth=linewidth)
plt.plot(x, np.array(sp_superglue_y), color='black', label='SuperPoint+SuperGlue',linewidth=linewidth)

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
plt.title("Viewpoint",fontdict={'family': 'Times New Roman', 'size': 25})

# 添加X轴和Y轴标签
plt.xlabel("Thresholds(px)",fontdict={'family': 'Times New Roman', 'size': 25})
plt.ylabel("MMA",fontdict={'family': 'Times New Roman', 'size': 25})

plt.legend(loc='best', framealpha=0,prop={'family': 'Times New Roman', 'size': 18})

plt.savefig('C:\\Users\\leewe\\Desktop\\工作\\论文图像\\5.11.png', dpi=600)
# 显示图表
plt.show()
