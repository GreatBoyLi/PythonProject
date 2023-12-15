import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LoFTR_y = \
    [0, 0.2852, 0.4900, 0.6053, 0.6733, 0.7049, 0.7231, 0.7445, 0.7705, 0.7858, 0.7976]

MatchFormer_y = \
    [0, 0.2605, 0.4652, 0.5751, 0.6523, 0.6852, 0.7152, 0.7345, 0.7555, 0.7678, 0.7723]

SwinMatcher_y = \
    [0, 0.2902, 0.4932, 0.6072, 0.6773, 0.7093, 0.7352, 0.7545, 0.7755, 0.7878, 0.7986]

sift_superglue_y = \
    [0, 0.1156, 0.2032, 0.2735, 0.3665, 0.4221, 0.4489, 0.4641, 0.4798, 0.4879, 0.4949]

orb_gms_y = \
    [0, 0.1010, 0.1921, 0.2598, 0.3450, 0.4057, 0.4378, 0.4533, 0.4650, 0.4713, 0.4750]

sp_pointcn_y = \
    [0, 0.1310, 0.2221, 0.3010, 0.3999, 0.4651, 0.4949, 0.5002, 0.5065, 0.5103, 0.5126]

sp_superglue_y = \
    [0, 0.1604, 0.3025, 0.4275, 0.4993, 0.5518, 0.5756, 0.5872, 0.5995, 0.6053, 0.6089]

from curve import get_pose_error_curve

plt.subplots(figsize=(7, 10))

# sp_superglue_y = get_pose_error_curve(0.1616,0.3381,0.5184)
linewidth = 1.5

# sp_superglue_y = get_pose_error_curve(0.1616,0.3381,0.5184)
# 创建曲线图
plt.plot(x, np.array(LoFTR_y) * 100, color='cyan',linewidth=linewidth, label='LoFTR')
plt.plot(x, np.array(MatchFormer_y) * 100, color='red', linewidth=linewidth,label='MatchFormer')
plt.plot(x, np.array(SwinMatcher_y) * 100, color='green', linewidth=linewidth,label='SwinMatcher')
plt.plot(x, np.array(sift_superglue_y) * 100, color='blue', linewidth=linewidth,label='SIFT+SuperGlue')
plt.plot(x, np.array(orb_gms_y) * 100, color='orange', linewidth=linewidth,label='ORB+GMS')
plt.plot(x, np.array(sp_pointcn_y) * 100, color='purple', linewidth=linewidth,label='SuperPoint+PointCN')
plt.plot(x, np.array(sp_superglue_y) * 100, color='black', linewidth=linewidth,label='SuperPoint+SuperGlue')

plt.xticks([0,1,2, 3,4, 5, 6,7,8, 9,10])
plt.yticks([0, 10,20,30, 40,50, 60,70, 80,90])
plt.grid(True)
plt.xlim(1,10)  # X轴的界限从0开始
plt.ylim(10,85)

ax = plt.gca()
# 不显示上面和右边的边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.tick_params(labelsize=20)
# 添加标题
plt.title("Homograph estimation AUC", fontdict={'family': 'Times New Roman', 'size': 25})

# 添加X轴和Y轴标签
plt.xlabel("thresholds of pose error(px)",fontdict={'family': 'Times New Roman', 'size': 25})
plt.ylabel("AUC(%)",fontdict={'family': 'Times New Roman', 'size': 25})

plt.legend(loc='best', framealpha=0,prop={'family': 'Times New Roman', 'size': 18})

plt.savefig('C:\\Users\\leewe\\Desktop\\工作\\论文图像\\5.4.png', dpi=600)
# 显示图表
plt.show()
