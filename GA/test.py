import numpy as np
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.detectors import build_detector
from pcdet.utils import common_utils
#
#
def load_perturbation(perturbation_file):
    # 从txt文件中读取扰动点坐标
    # 返回 N * 3 的点云数组
    with open(perturbation_file, 'r') as file:
        lines = file.readlines()
        perturbation = np.array([list(map(float, line.strip().split(' '))) for line in lines])
    return perturbation
perturbation_file = '随机扰动点.txt'
# print(load_perturbation(perturbation_file))
perturbation=load_perturbation(perturbation_file)
print(perturbation)
# def get_fitness(perturbation):
#     # 将扰动点加入基础点集，得到新的点集
#     # 使用pointrcnn进行目标检测，返回所有类别的平均置信度
#     content1 = []
#     with open('/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/bintotxt/1.txt', 'r') as f:
#         content1 = f.readlines()
#     # 合并两个txt文件的内容
#     perturbed_points = content1 + perturbation
#     # 将合并后的内容存储到一个新的txt文件中
#     with open('result.txt', 'w') as f:
#         f.writelines(perturbed_points)
#     return perturbed_points
# print('---------------------------')
# print(get_fitness(perturbation))
content1 = []
with open('/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/bintotxt/1.txt', 'r') as f:
    content1 = f.readlines()
content2 = []
with open('随机扰动点.txt', 'r') as f:
    content2 = f.readlines()
# 合并两个txt文件的内容
perturbed_points = content1 + content2
# 将合并后的内容存储到一个新的txt文件中
with open('perturbed_and_raw.txt', 'w') as f:
    f.writelines(perturbed_points)
print('---------------------------')
print(perturbed_points)
