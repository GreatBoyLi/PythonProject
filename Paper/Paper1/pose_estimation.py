import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
LoFTR_y = \
    [0, 0.0112, 0.0531, 0.1102, 0.1675, 0.2195, 0.2659, 0.3076, 0.3443, 0.3765, 0.4051, 0.4307, 0.4539, 0.4748, 0.4938,
     0.5110, 0.5267, 0.5410, 0.5541, 0.5661, 0.5773]
MatchFormer_y = \
    [0, 0.0090, 0.0425, 0.0952, 0.1500, 0.2010, 0.2484, 0.2906, 0.3273, 0.3608, 0.3910, 0.4177, 0.4416, 0.4631, 0.4824,
     0.5004, 0.5169, 0.5323, 0.5466, 0.5598, 0.5721]

SwinMatcher_y = \
    [0, 0.0110, 0.0531, 0.1099, 0.1673, 0.2187, 0.2662, 0.3081, 0.3449, 0.3772, 0.4061, 0.4316, 0.4550, 0.4752, 0.4940,
     0.5115, 0.5269, 0.5413, 0.5596, 0.5704, 0.5778]

sift_superglue_y = \
    [0, 0.0034, 0.0162, 0.0336, 0.0512, 0.0671, 0.0884, 0.1050, 0.1213, 0.1420, 0.1570, 0.1795, 0.1993, 0.2184, 0.2309,
     0.2416, 0.2525, 0.2631, 0.2741, 0.2809, 0.2867]

orb_gms_y = \
    [0, 0.0026, 0.0125, 0.0260, 0.0390, 0.0549, 0.0732, 0.0905, 0.1059, 0.1195, 0.1363, 0.1496, 0.1703, 0.1859, 0.1964,
     0.20510, 0.2183, 0.2251, 0.2363, 0.2457, 0.2532]

sp_pointcn_y = \
    [0, 0.0058, 0.0275, 0.0572, 0.0869, 0.1190, 0.1503, 0.1856, 0.2164, 0.2507, 0.2767, 0.3059, 0.3255, 0.3405, 0.3542,
     0.3665, 0.3778, 0.3880, 0.3974, 0.4060, 0.4141]
sp_superglue_y = \
    [0, 0.0082, 0.0391, 0.0811, 0.1231, 0.1696, 0.2119, 0.2547, 0.2893, 0.3242, 0.3551, 0.3848, 0.4076, 0.4263, 0.4434,
     0.4589, 0.4729, 0.4858, 0.4976, 0.5083, 0.5184]

from curve import get_pose_error_curve

plt.subplots(figsize=(7, 10))

# sp_superglue_y = get_pose_error_curve(0.1616,0.3381,0.5184)
linewidth = 2
linestyle = '-'
# 创建曲线图
plt.plot(x, np.array(sp_superglue_y) * 100, color='black', linewidth=linewidth, label='SuperPoint+SuperGlue',linestyle=linestyle)
plt.plot(x, np.array(sp_pointcn_y) * 100, color='purple', linewidth=linewidth, label='SuperPoint+PointCN',linestyle=linestyle)
plt.plot(x, np.array(sift_superglue_y) * 100, color='blue', linewidth=linewidth, label='SIFT+SuperGlue',linestyle=linestyle)
plt.plot(x, np.array(LoFTR_y) * 100, color='cyan', linewidth=linewidth, label='LoFTR',linestyle=linestyle)
plt.plot(x, np.array(MatchFormer_y) * 100, color='red', linewidth=linewidth, label='MatchFormer',linestyle=linestyle)
plt.plot(x, np.array(SwinMatcher_y) * 100, color='green', linewidth=linewidth, label='SwinMatcher',linestyle=linestyle)
plt.plot(x, np.array(orb_gms_y) * 100, color='orange', linewidth=linewidth, label='ORB+GMS',linestyle=linestyle)
plt.grid(True)

plt.xticks([0,2,4,6,8, 10,12,14, 16,18, 20])
plt.yticks([0, 10,20,30,40,50,60])
plt.xlim(2,20)  # X轴的界限从0开始
plt.ylim(0,65)

ax = plt.gca()
# 不显示上面和右边的边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.tick_params(labelsize=20)

# 添加标题
plt.title("Pose estimation AUC",fontdict={'family': 'Times New Roman', 'size': 25})

# 添加X轴和Y轴标签
plt.xlabel("thresholds of pose error(°)", fontdict={'family': 'Times New Roman', 'size': 25})
plt.ylabel("AUC(%)", fontdict={'family': 'Times New Roman', 'size': 25})

plt.legend(loc='best', framealpha=0,prop={'family': 'Times New Roman', 'size': 18})

plt.savefig('C:\\Users\\leewe\\Desktop\\工作\\论文图像\\5.1.png', dpi=600)
# 显示图表
plt.show()
