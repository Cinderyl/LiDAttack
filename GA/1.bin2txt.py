import os
import struct
import numpy as np

"""
功能：
	python 批量实现点云.bin文件转换为.txt文件
输入：
	.bin文件根目录,保存txt文件的根目录

输出: 
	x,y,z,i格式的txt文件，分割符号为空格

"""


def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2], point[3]])
    return np.asarray(pc_list, dtype=np.float32)


def bin2txt(data, save_name):
    np.savetxt((save_name + '.txt'), data, fmt="%.6f", delimiter=" ")

def main():
    root_dir = '/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/velodyne'  # 点云.bin文件路径
    save_path = '/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/bintotxt'  # .txt文件保存路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        savename = os.path.join(save_path, str(i + 1))
        data = read_velodyne_bin(filename)
        bin2txt(data, savename)


if __name__ == "__main__":
    main()