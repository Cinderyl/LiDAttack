import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

# 输入的三维点
points = [
    [3.347895812988281250, -3.263081789016723633, -1.625078320503234863],
    [3.350003051757812500, -2.999969005584716797, -1.608140110969543457],
    [3.343355941772460938, -3.313758611679077148, -1.499996066093444824],
    [3.350000381469726562, -3.500001907348632812, -1.499991059303283691],
    [3.347271728515625000, -3.328490734100341797, -1.497986316680908203],
    [3.347274398803710938, -3.265915155410766602, -2.031375885009765625],
    [3.350000381469726562, -3.500002384185791016, -1.499991059303283691],
    [3.343355941772460938, -3.313758611679077148, -1.499996066093444824],
    [3.350003051757812500, -2.999969005584716797, -1.608140110969543457],
    [3.347895812988281250, -3.263081789016723633, -1.625078320503234863]
]

# 从点中选择三个点来生成所有可能的三角形
triangles = list(itertools.combinations(points, 3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用Poly3DCollection绘制多面体的面
ax.add_collection3d(Poly3DCollection(triangles, facecolor='cyan', linewidths=1, edgecolors='r', alpha=.25))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 根据点的坐标设置坐标轴范围
ax.set_xlim([min(p[0] for p in points), max(p[0] for p in points)])
ax.set_ylim([min(p[1] for p in points), max(p[1] for p in points)])
ax.set_zlim([min(p[2] for p in points), max(p[2] for p in points)])

plt.show()