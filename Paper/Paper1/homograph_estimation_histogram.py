import matplotlib.pyplot as plt

# # 数据
# categories = ['SIFT+\nSuperGlue', 'ORB+\nGMS', 'SP+\nPointCN', 'SP+\nSuperGlue', 'LoFTR', 'MatchForer','Swin-\nMatcher']
# values = [76.85, 74.65, 77.22, 82.18, 89.17, 88.22, 90.82]
#
# # 创建柱状图
# plt.bar(categories, values,color=['blue', 'orange', 'purple', 'black', 'cyan','red','green'])
#
# # 添加标题和轴标签
# plt.title('Homograph estimation percision')
# # plt.xlabel('Category')
# plt.ylabel('Precision')
#
# # plt.grid(True, axis='y', linestyle='-', linewidth=0.9, color='black', alpha=0.7)  # 只在y轴方向显示网格
#
# plt.savefig('C:\\Users\\leewe\\Desktop\\工作\\论文图像\\5.5.png', dpi=600)
#
# # 显示图表
# plt.show()

# 数据
categories = ['SIFT+\nSuperGlue', 'ORB+\nGMS', 'SP+\nPointCN', 'SP+\nSuperGlue', 'LoFTR', 'MatchForer','Swin-\nMatcher']
values = [81.56, 79.01, 86.51, 90.75, 97.29, 96.17, 97.41]

# 创建柱状图
plt.bar(categories, values,color=['blue', 'orange', 'purple', 'black', 'cyan','red','green'])

# 添加标题和轴标签
plt.title('Homograph estimation recall')
# plt.xlabel('Category')
plt.ylabel('Recall')

# plt.grid(True, axis='y', linestyle='-', linewidth=0.9, color='black', alpha=0.7)  # 只在y轴方向显示网格

plt.savefig('C:\\Users\\leewe\\Desktop\\工作\\论文图像\\5.6.png', dpi=600)

# 显示图表
plt.show()
