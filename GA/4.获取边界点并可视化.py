import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# 读取TXT格式的三维点云
def read_point_cloud(filename):
    with open(filename, 'r') as f:
        data = np.array([[float(num) for num in line.split()] for line in f])
    return data

# 获取三维点云的凸包边界
def get_convex_hull(point_cloud):
    hull = ConvexHull(point_cloud)
    boundary_points = point_cloud[hull.vertices]
    return boundary_points

# 可视化点云
def visualize_point_cloud(point_cloud_1, point_cloud_2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], point_cloud_1[:, 2], c='blue')
    ax.scatter(point_cloud_2[:, 0], point_cloud_2[:, 1], point_cloud_2[:, 2], c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# 读取三维点云
point_cloud = read_point_cloud('提取.txt')

# 获取点云的边界点
boundary_points = get_convex_hull(point_cloud)
np.savetxt(('边界点.txt'), boundary_points, fmt="%.6f", delimiter=" ")
# 可视化原始点云和边界点云
visualize_point_cloud(point_cloud, boundary_points)
