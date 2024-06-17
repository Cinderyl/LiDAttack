import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.detectors import build_detector
# from pcdet.utils import get_pcds_from_nuscenes_infos
from pcdet.config import cfg, cfg_from_yaml_file
import argparse
from pathlib import Path

# 定义扰动优化类
class AdversarialOptimizer(object):
    def __init__(self, base_points_file, perturbation_file, pointrcnn_config_file, fitness_threshold=0.5,
                 max_iterations=100, population_size=10, mutation_rate=0.1):
        self.base_points_file = base_points_file
        self.perturbation_file = perturbation_file
        self.fitness_threshold = fitness_threshold
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        # 加载基础点集
        with open(base_points_file, 'r') as f:
            self.base_points = np.loadtxt(f)

        # 加载扰动点集
        with open(perturbation_file, 'r') as f:
            self.perturbation = np.loadtxt(f)

        # 转换扰动点集格式
        if len(self.perturbation.shape) == 1:  # 对于只有一个扰动点的情况，需要将其转换为一个二维数组，以便与基础点集进行拼接
            self.perturbation = self.perturbation.reshape(1, self.perturbation.shape[0])

        # 加载目标检测算法pointrcnn
        cfg_from_yaml_file(pointrcnn_config_file)
        self.pointrcnn_detector = build_detector()
        self.pointrcnn_detector.load_weights(cfg.MODEL_DIR)

    # 定义适应度函数
    def get_fitness(self, perturbation_points):
        # 将扰动点加入基础点集，得到新的点集
        new_points = np.concatenate((self.base_points, perturbation_points), axis=0)

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

        # 返回所有类别的平均置信度
        return np.mean(confidence)

    # 定义选择操作函数
    def selection(self, fitnesses, num_parents):
        # 选择前num_parents个适应度最高的个体作为父代
        parents = np.zeros((num_parents, self.perturbation.shape[1]))
        for i in range(num_parents):
            index = np.argmax(fitnesses)
            parents[i, :] = self.population[index, :]
            fitnesses[index] = -1e9

        return parents

    # 定义交叉操作函数
    def crossover(self, parents):
        # 随机选择父代进行交叉
        offspring = np.zeros((self.population_size - parents.shape[0], self.perturbation.shape[1]))
        num_offspring = offspring.shape[0]

        for i in range(num_offspring):
            parent1_index = np.random.randint(0, parents.shape[0])
            parent2_index = np.random.randint(0, parents.shape[0])

            while parent2_index == parent1_index:
                parent2_index = np.random.randint(0, parents.shape[0])

            # 随机选择交叉点
            crossover_point = np.random.randint(1, self.perturbation.shape[1] - 1)

            # 父代自适应交叉
            if parents[parent1_index, -1] > parents[parent2_index, -1]:
                offspring[i, :crossover_point] = parents[parent1_index, :crossover_point]
                offspring[i, crossover_point:] = parents[parent2_index, crossover_point:]
            else:
                offspring[i, :crossover_point] = parents[parent2_index, :crossover_point]
                offspring[i, crossover_point:] = parents[parent1_index, crossover_point:]

        return offspring

    # 定义变异操作函数
    def mutation(self, offspring_crossover):
        # 对于每个个体，每个位置进行一定概率的变异
        num_mutations = int((self.mutation_rate * self.perturbation.shape[1]) * (self.population_size - 1))
        for i in range(1, self.population_size):
            mutation_indices = np.random.choice(self.perturbation.shape[1], num_mutations, replace=False)
            for j in mutation_indices:
                offspring_crossover[i, j] += np.random.randn() * 0.05

        return offspring_crossover

    # 定义遗传算法迭代函数
    def evolve(self):
        # 初始化种群
        self.population = np.vstack([self.perturbation] * self.population_size)

        # 迭代优化
        fitnesses = np.zeros(self.population_size)
        for i in range(self.max_iterations):
            # 计算适应度值
            for j in range(self.population_size):
                fitnesses[j] = self.get_fitness(self.population[j, :])
                self.population[j, -1] = fitnesses[j]

            # 输出每次迭代的适应度值
            print('Iteration:', i + 1, 'Best fitness:', np.max(fitnesses))

            # 终止条件：适应度值达到阈值或达到最大迭代次数
            if np.max(fitnesses) >= self.fitness_threshold or i == self.max_iterations - 1:
                break

            # 进行遗传操作
            parents = self.selection(fitnesses, num_parents=2)
            offspring_crossover = self.crossover(parents)
            offspring_mutation = self.mutation(offspring_crossover)
            self.population[1:, :] = offspring_mutation

        # 输出最优的扰动点
        index = np.argmax(fitnesses)
        best_perturbation = self.population[index, :]
        print('Best perturbation:', best_perturbation)

        # 输出最终的适应度值和目标检测结果
        best_fitness = self.get_fitness(best_perturbation)
        print('Best fitness:', best_fitness)
        self.visualize(self.base_points, best_perturbation)

    # 定义可视化函数，以便查看最终的优化效果
    def visualize(self, base_points, perturbation):
        new_points = np.concatenate((base_points, perturbation.reshape(1, -1)), axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_points)
        o3d.visualization.draw_geometries([pcd])


import os
import numpy as np
from pyquaternion import Quaternion


class KittiDataset:
    def __init__(self, root_path):
        self.root_path = root_path

    def __getitem__(self, index):
        sequence_id = '0000'
        velo_filename = os.path.join(self.root_path, sequence_id, 'velodyne', f'{index:06}.bin')
        with open(velo_filename, 'rb') as f:
            velo_data = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3]

        calib_filename = os.path.join(self.root_path, sequence_id, 'calib', f'{index:06}.txt')
        with open(calib_filename, 'r') as f:
            P0, P1, P2, _, _, _, _, R0_rect, Tr_velo_to_cam, _, _ = [line.split()[1:] for line in f.readlines()]
            P0, P1, P2 = np.array(P0, dtype=np.float32), np.array(P1, dtype=np.float32), np.array(P2, dtype=np.float32)
            R0_rect, Tr_velo_to_cam = np.array(R0_rect, dtype=np.float32).reshape(3, 3), np.array(Tr_velo_to_cam,
                                                                                                  dtype=np.float32).reshape(
                3, 4)
            T3 = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, -1)
            Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, T3], axis=0)
            R_cam_to_rect = np.eye(4, dtype=np.float32)
            R_cam_to_rect[:3, :3] = R0_rect
            R_cam_to_rect_inv = np.linalg.inv(R_cam_to_rect)
        velo_rect_data = np.concatenate([velo_data, np.ones_like(velo_data[:, :1])], axis=1) @ np.linalg.inv(
            R_cam_to_rect) @ Tr_velo_to_cam.T
        velo_rect_data = velo_rect_data[:, :3] / velo_rect_data[:, 3:]

        return {
            'points': velo_rect_data.astype(np.float32),
            'index': index,
            'frame_id': sequence_id
        }

    def __len__(self):
        return 180

# 测试代码
if __name__ == '__main__':
    # 定义文件路径
    base_points_file = os.path.join('/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/velodyne', '000446.bin')
    perturbation_file = os.path.join('/data0/benke/ldx/openpcdet37/OpenPCDet/GA', '随机扰动点.txt')
    # pointrcnn_config_file = '/data0/benke/ldx/openpcdet37/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml'

    # 加载基础点集
    with open(base_points_file, 'rb') as f:
        base_points = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)

    # 加载目标检测算法pointrcnn
    def parse_config():
        parser = argparse.ArgumentParser(description='OpenPCDet Grad-CAM')
        parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointrcnn.yaml',
                            help='specify the config for demo')
        parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
        parser.add_argument('--data_path', type=str, default='../data/kitti', help='specify the path to data')
        parser.add_argument('--split', type=str, default='val', help='specify the data split')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
        parser.add_argument('--save_dir', type=str, default='outputs/gradcam', help='directory to save results')
        parser.add_argument('--vis', action='store_true', default=True, help='visualize results')
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        args = parser.parse_args()
        return args


    args = parse_config()

    # load config
    cfg_from_yaml_file(args.cfg_file, cfg)
    # cfg_from_yaml_file(pointrcnn_config_file,cfg)
    detector = build_detector()

    # 加载Kitti数据集
    # dataset = KittiDataset(root_path='data/kitti')
    # data_infos = dataset.get_sequence_data_infos('0000')
    # pcds = get_pcds_from_nuscenes_infos(
    #     dataset, data_infos, cfg.MODEL.INPUT_SPEC, num_worker_threads=4
    # )
    dataset = KittiDataset(root_path='/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object')
    pcds = []
    for i in range(len(dataset)):
        data = dataset[i]
        pcds.append(data['points'])

    pcds = np.concatenate(pcds, axis=0)


    # 显示原始点云数据
    o3d.visualization.draw_geometries(pcds)

    # 用pointrcnn进行目标检测，得到每个目标物体的类别和置信度
    detection_dict = {}
    detection_dict['points'] = pcds[0]
    _, det_annos = detector.forward_test([detection_dict])
    confidence = []
    for anno in det_annos:
        for obj in anno['annos']:
            if obj['score'] >= 0.1:
                confidence.append(obj['score'])
    confidence = np.array(confidence)

    # 显示目标物体的置信度分布情况
    plt.hist(confidence, bins=50)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.show()

    # 定义扰动优化器实例，并调用其优化函数
    optimizer = AdversarialOptimizer(
        base_points_file, perturbation_file, pointrcnn_config_file,
        fitness_threshold=0.3, max_iterations=100, population_size=20, mutation_rate=0.1)

    optimizer.evolve()