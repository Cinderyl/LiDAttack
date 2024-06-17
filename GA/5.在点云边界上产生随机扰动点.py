import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取TXT格式的三维点云
def read_point_cloud(filename):
    with open(filename, 'r') as f:
        data = np.array([[float(num) for num in line.split()] for line in f])
    return data

# 在点云周围添加扰动点
def add_noise(point_cloud, std=0.1):
    noise = np.random.normal(0, std, size=point_cloud.shape)
    noisy_cloud = point_cloud + noise
    return noisy_cloud

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
point_cloud = read_point_cloud('边界点.txt')

# 在点云周围添加扰动点
noisy_cloud = add_noise(point_cloud)
np.savetxt(('随机扰动点.txt'), noisy_cloud, fmt="%.6f", delimiter=" ")
# 可视化原始点云和添加扰动后的点云
visualize_point_cloud(point_cloud, noisy_cloud)
