import numpy as np
import os
# a=[1,2,3,4]
# print(a[1:3])
# 路径设置
dataset_path = '/data0/benke/ldx/openpcdet37/OpenPCDet/data/kitti/training'
velodyne_path = os.path.join(dataset_path, 'velodyne')
label_path = os.path.join(dataset_path, 'label_2')

# 目标物体设置
target_object = 'Car'

# 遍历所有样本
for sample_id in range(0, 7481):
    # 读取对应的txt文件，提取边界框的信息
    label_file = os.path.join(label_path, '%06d.txt' % sample_id)
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [line.strip().split(' ') for line in lines]
    # 遍历所有物体边界框
    i=0
    for obj in objects:
        if obj[0] == target_object:
            # 提取目标物体的四个坐标和朝向角度
            h,w,l, x, y, z, yaw = map(float, obj[8:15])

            # 计算目标物体在点云中的索引
            point_cloud_file = os.path.join(velodyne_path, '%06d.bin' % sample_id)
            point_cloud = np.fromfile(point_cloud_file, dtype=np.float32).reshape(-1, 4)
            x_range = (point_cloud[:, 0] > (x - l / 2)) & (point_cloud[:, 0] < (x + l / 2))
            y_range = (point_cloud[:, 1] > (y - w / 2)) & (point_cloud[:, 1] < (y + w / 2))
            z_range = (point_cloud[:, 2] > (z - h)) & (point_cloud[:, 2] < z)
            indices = x_range & y_range & z_range

            # 提取目标物体的点云数据
            target_cloud = point_cloud[indices, :]


            # 保存点云数据
            # target_cloud_file = os.path.join(dataset_path, 'target_cloud', '%06d_%d.bin' % (sample_id, int(obj[1])))
            # target_cloud.tofile(target_cloud_file)
            object_file = '/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/提取物体的点云数据/{}.txt'.format(obj)
            np.savetxt(object_file, target_cloud, delimiter=' ', fmt='%.5f')
            i = i + 1