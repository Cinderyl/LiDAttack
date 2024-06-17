import numpy as np
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.detectors import build_detector
from pcdet.utils import common_utils
import open3d as o3d
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.detectors import build_detector
# from pcdet.utils import get_pcds_from_nuscenes_infos
from pcdet.config import cfg, cfg_from_yaml_file

class GeneticAlgorithm:
    def __init__(self, base_point_cloud, perturbation_file, detector_cfg ):
        self.base_cloud = base_point_cloud
        self.perturbation = self.load_perturbation(perturbation_file)
        self.detector_cfg = detector_cfg
        self.detector = build_detector(self.detector_cfg).cuda()

    def load_perturbation(self, perturbation_file):
        # 从txt文件中读取扰动点坐标
        # 返回 N * 3 的点云数组
        with open(perturbation_file, 'r') as file:
            lines = file.readlines()
            perturbation = np.array([list(map(float, line.strip().split(' '))) for line in lines])
        return perturbation


    def get_fitness(self, perturbation):
        # 将扰动点加入基础点集，得到新的点集
        # 使用pointrcnn进行目标检测，返回所有类别的平均置信度
        content1 = []
        with open('/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/bintotxt/1.txt', 'r') as f:
            content1 = f.readlines()
        content2 = []
        with open('随机扰动点.txt', 'r') as f:
            content2 = f.readlines()
        # 合并两个txt文件的内容
        new_points = content1 + content2
        # 将合并后的内容存储到一个新的txt文件中
        with open('perturbed_and_raw.txt', 'w') as f:
            f.writelines(new_points)
        print('---------------------------')
        print(new_points)


        # 加载目标检测算法pointrcnn
        cfg_from_yaml_file(pointrcnn_config_file)
        self.pointrcnn_detector = build_detector()
        self.pointrcnn_detector.load_weights(cfg.MODEL_DIR)
        # 转换点集格式
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(new_points)

        # 使用pointrcnn进行目标检测，返回所有类别的置信度
        detection_dict = {}
        detection_dict['points'] = point_cloud
        _, det_annos = self.pointrcnn_detector.forward_test([detection_dict])
        confidence = []
        for anno in det_annos:
            for obj in anno['annos']:
                if obj['score'] >= 0.1:
                    confidence.append(obj['score'])
        confidence = np.array(confidence)



        pass

    def select(self, fitness):
        # 根据适应度函数对个体进行选择，可以使用轮盘赌算法、锦标赛算法等
        pass

    def mutate(self, selected_pop):
        # 对选中的个体进行变异操作，比如随机删除或添加扰动点
        pass

    def crossover(self, selected_pop):
        # 对选中的个体进行交叉操作，比如随机交换相邻扰动点
        pass

    def run(self, pop_size, max_iter, criterion):
        # 初始化种群
        # 迭代
            # 计算适应度函数
            # 进行选择、变异、交叉操作
            # 判断截止条件
        # 输出最佳扰动点组合
        pass

if __name__ == '__main__':
    # 读取原始点云数据并提取关键点
    dataset = KittiDataset(anno_file, root_path)
    point_cloud = dataset.get_lidar(0)
    selected_indices = common_utils.random_sampling(point_cloud.shape[0], 2048)
    base_cloud = point_cloud[selected_indices]

    # 初始化遗传算法
    perturbation_file = 'perturbation.txt'
    detector_cfg = 'pointrcnn.yaml'
    ga = GeneticAlgorithm(base_cloud, perturbation_file, detector_cfg)

    # 运行遗传算法
    pop_size = 20
    max_iter = 100
    criterion = {'confidence': 0.3, 'max_iter': 100}
    best_perturbation = ga.run(pop_size, max_iter, criterion)

    # 将扰动点加入原始点云并进行目标检测
    point_cloud += best_perturbation
    input_dict = {'points': point_cloud}
    result_dict = ga.detector(input_dict)

    # 可视化结果
    pass