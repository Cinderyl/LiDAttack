import numpy as np

# # 读取原始文件
# with open("G:\Desktop\点云\p1-person- Cloud.txt", "r") as f:
#     lines = f.readlines()
#
# # 将每一行的数据转换为列表，并只保留前四列
# data = [list(map(float, line.strip().split()[:4])) for line in lines]
#
#
# # 将数据转换为numpy数组
# data_array = np.array(data)
#
# # 保存到新文件
# np.savetxt("G:\Desktop\点云\p1-person- Cloud_4.txt", data_array)
# import numpy as np
#
# 读取原始txt文件
# data = np.loadtxt('/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/八个顶点/test/p1-allcloud.txt')
#
# # 取出前四列
# result = data[:, :4]
#
# # 将第四列的元素乘以0.1
# result[:, 3] = result[:, 3] * 10
#
# # 保存到新的txt文件
# np.savetxt('/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/八个顶点/test/000001.txt', result, delimiter=' ', fmt='%f')
#
data = np.loadtxt('/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/提取的点云/Pedestrian/Easy_test//000001.txt')
# 取出前四列
result = data[:, :4]

# 将第四列的元素乘以0.1
result[:, 3] = result[:, 3] * 10

# 保存到新的txt文件
np.savetxt('/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/提取的点云/Pedestrian/Easy_test//000001.txt', result, delimiter=' ', fmt='%f')


# import os
# import struct
#
# file_number = "000001"  # 用实际的文件编号替换
# txt_file_path = "/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/八个顶点/test/000001.txt"  # 用实际的txt文件路径替换
# bin_file_path = os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/data/kitti/training/velodyne_test", f"{file_number}.bin")
#
# with open(txt_file_path, 'r') as txt_file:
#     with open(bin_file_path, 'wb') as bin_file:
#         for line in txt_file:
#             x, y, z, intensity = map(float, line.strip().split())
#             bin_file.write(struct.pack('ffff', x, y, z, intensity))