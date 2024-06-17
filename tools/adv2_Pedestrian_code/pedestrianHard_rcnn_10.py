import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import time
time_start = time.time()  # 记录开始时间
import argparse
import glob
import math
from pathlib import Path
import random
import struct
# try:
#     import open3d
#     from tools.visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import tools.mayavi.mlab as mlab
#     from tools.visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
max_iterations=1000
pc=0.5
mutation_rate=0.01
num_elites=10
tournament_size=4
mindist=1
# fitness_threshold=0.5
# 文件夹路径，将其替换为您实际的文件夹路径
folder_path1 = "/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/八个顶点/Pedestrian/Hard"
folder_path2="/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/input_data输入/提取的点云/Pedestrian/Hard"
# 获取文件夹中所有文件的文件名列表
file_names = os.listdir(folder_path2)
# 过滤出所有的txt文件
txt_files = [file_name for file_name in file_names if file_name.endswith(".txt")]


def distance(point1, point2):
    # 计算两个点之间的欧几里得距离
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def chamfer_distance(set1, set2):
    # 计算两个点集之间的倒角距离
    distances = []
    for point1 in set1:
        # 计算点集A中每个点到点集B中所有点的距离的最小值
        min_distance = min([distance(point1, point2) for point2 in set2])
        distances.append(min_distance)
    for point2 in set2:
        # 计算点集B中每个点到点集A中所有点的距离的最小值
        min_distance = min([distance(point2, point1) for point1 in set1])
        distances.append(min_distance)
    # 取平均值作为倒角距离
    return sum(distances) / len(distances)

def euclidean_distance(point1, point2):
    # 计算两个点之间的欧氏距离
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def calculate_distances(points):
    # 计算10个点之间的距离
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = euclidean_distance(points[i], points[j])
            distances.append(distance)
    distances = sum(distances)
    return distances


def fitness_function(pred_scores, pred_labels, points, individual):
    individual = individual[:, 0:3]
    distances = calculate_distances(individual)
    beta = 10
    #    Z = chamfer_distance(points, individual)
    # 0-0.5识别为目标物体，且置信度大于0.3
    if (pred_labels == torch.tensor(2).to(device)):
        return 1 - pred_scores
    # 0.485-1没有识别到该物体或者识别到该物体了但是置信度小于0。3
    elif (pred_labels == torch.tensor(0).to(device)):
        return (1 - pred_scores) + beta * (1 / (1 + distances)) + 1
    # 0.515-1
    # 识别成了其他物体
    else:
        return pred_scores + beta * (1 / (1 + distances)) + 3

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config(file_number):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../cfgs/kitti_models/pointrcnn1.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default= os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/data/kitti/training/velodyne", f"{file_number}.bin"),
                        help='specify the point cloud data file or directory')
    parser.add_argument('--per_data_path', type=str, default=os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/中间bin/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard", f"{file_number}.bin"),
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/output/cfgs/kitti_models/pointrcnn/default/ckpt/checkpoint_epoch_51.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg
# 我要实现的功能：输入扰动点之后，强度我都设置为5，扰动点加上原始点，就得到新的点，转为.bin文件，输入到模型里面得到置信度和预测类别
def attack_GA1(points,perturbations,args,cfg,logger,first_pred_boxes,file_number,model):
    new_points = np.concatenate((points, perturbations), axis=0)
    with open(os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/中间bin/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard", f"{file_number}.bin"), 'wb') as f:
        # 依次将每个点的四个值转换为二进制，并写入文件中
        for point in new_points:
            x, y, z, intensity = point
            f.write(struct.pack('ffff', x, y, z, intensity))
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG1, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.per_data_path), ext=args.ext, logger=logger
    )
    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    # model.cuda()
    # model.eval()
    data_dict = demo_dataset[0]
    per_points = data_dict['points']
    per_points_xyz = per_points[:, :3]


    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    pred_boxes = pred_dicts[0]['pred_boxes']

    if (len(pred_boxes)!=0):
        m=0
        for boxes in pred_boxes:
            dx=boxes[0]-first_pred_boxes[0]
            dy=boxes[1]-first_pred_boxes[1]
            dz=boxes[2]-first_pred_boxes[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            pred_scores = pred_dicts[0]['pred_scores'][m]
            pred_labels = pred_dicts[0]['pred_labels'][m]
            m=m+1
            # 说明没有检测到
            if dist>2:
                pred_scores = torch.tensor(1.2)
                # pred_labels=torch.tensor(4)
                # pred_scores=pred_scores.to(device)
                # pred_labels =pred_labels.to(device)
                continue
            # 检测到了，可能时源标签，也可能是其他的
            else:

                return pred_scores,pred_labels
        if(pred_scores==torch.tensor(1.2)):
            pred_scores=torch.tensor(0)
            pred_labels=torch.tensor(0)
            pred_scores = pred_scores.to(device)
            pred_labels = pred_labels.to(device)
            return pred_scores,pred_labels


    else:
        pred_scores = torch.tensor(0)
        pred_labels = torch.tensor(0)
        pred_scores = pred_scores.to(device)
        pred_labels = pred_labels.to(device)
    return pred_scores,pred_labels

def roulette_selection(population, fitness_values):
    # 计算适应度总和
    fitness_sum = sum(fitness_values)
    # 计算每个个体的选择概率
    selection_prob = [fitness / fitness_sum for fitness in fitness_values]
    # 计算累积概率
    cum_prob = [sum(selection_prob[:i+1]) for i in range(len(selection_prob))]
    # 选择新种群的父代
    new_population = []
    num_selected=20
    # for i in range(len(population)):
    for i in range(num_selected):
        # 生成随机数
        r = random.random()
        # 根据随机数选择个体
        for j in range(len(cum_prob)):
            if r <= cum_prob[j]:
                new_population.append(population[j])
                break

    return new_population

def elitism_tournament_selection(population, fitness_values, num_elites, tournament_size,pre_scores,one_time_label):
    fitness=[]
    scores=[]
    label=[]
    # 根据适应度值对个体进行排序
    sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k], reverse=True)
    # 选择新种群的精英个体索引
    elite_indices = sorted_indices[:num_elites]
    for i in elite_indices:
        fitness.append(fitness_values[i])
        scores.append(pre_scores[i])
        label.append(one_time_label[i])
    new_population = [population[i] for i in elite_indices]

    while len(new_population) <20:
        tournament_candidates = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_candidates]
        winner_index = tournament_candidates[tournament_fitness.index(max(tournament_fitness))]
        fitness.append(fitness_values[winner_index])
        scores.append(pre_scores[winner_index])
        label.append(one_time_label[winner_index])
        new_population.append(population[winner_index])
    individual=new_population[np.argmax(fitness)]
    max_score=scores[np.argmax(fitness)]
    max_label=label[np.argmax(fitness)]

    return new_population,fitness,scores,individual,max_score,label,max_label
# 交叉
def one_point_crossover(population, pc,xmin,xmax,ymin,ymax,zmin,zmax):
    population_cross = []
    for individual in population:
        individual_3=individual[:,:3]
        binary=''
        for point in individual_3:
            binary_x = struct.pack('>f', point[0])
            binary_y = struct.pack('>f', point[1])
            binary_z = struct.pack('>f', point[2])
            binary_point = binary_x + binary_y + binary_z
            # 定义一个空字符串 `binary_str`，用来拼接二进制数字字符串。
            binary_str = ''
            for b in binary_point:
                # 表示将数字格式化为8位二进制数，不足8位则在左侧用0补齐。
                # 将上面得到的二进制数字符串拼接到 `binary_str` 变量中。
                binary_str += '{:08b}'.format(b)
            binary=binary_str+binary   #str,表示某一行的二进制
        # print(len(binary))
        population_cross.append([binary])#将一个扰动样本的所有行拼接起来
    # print(len(population_cross))         #20行
    # print('交叉之前',population_cross)

    # 定义新种群列表
    new_population = []

    # 遍历父代个体
    for i in range(0, len(population_cross), 2):
        new_individual1 = ''
        new_individual2 = ''
        crossover_point = random.randint(1, len(population_cross[i][0]) - 1)  # 随机选择交叉点
        for j in range(len(population_cross[i][0])):
            if j < crossover_point:
                new_individual1 += population_cross[i][0][j]
                new_individual2 += population_cross[i + 1][0][j]
            else:
                new_individual1 += population_cross[i + 1][0][j]
                new_individual2 += population_cross[i][0][j]


        T=True
        time_start1 = time.time()
        flag1=-1 #用作标记
        while(T):
            T=False
            for k in range(0, len(new_individual1), 96):  # 81120的长度96乘以845
                binary_x1 = new_individual1[k:k + 32]
                binary_y1 = new_individual1[k + 32:k + 64]
                binary_z1 = new_individual1[k + 64:k + 96]
                binary_x2 = new_individual2[k:k + 32]
                binary_y2 = new_individual2[k + 32:k + 64]
                binary_z2 = new_individual2[k + 64:k + 96]
                x1 = struct.unpack('>f', bytes.fromhex(hex(int(binary_x1, 2))[2:].zfill(8)))[0]
                y1 = struct.unpack('>f', bytes.fromhex(hex(int(binary_y1, 2))[2:].zfill(8)))[0]
                z1 = struct.unpack('>f', bytes.fromhex(hex(int(binary_z1, 2))[2:].zfill(8)))[0]
                x2 = struct.unpack('>f', bytes.fromhex(hex(int(binary_x2, 2))[2:].zfill(8)))[0]
                y2 = struct.unpack('>f', bytes.fromhex(hex(int(binary_y2, 2))[2:].zfill(8)))[0]
                z2 = struct.unpack('>f', bytes.fromhex(hex(int(binary_z2, 2))[2:].zfill(8)))[0]
                if (x1 < xmax and x1 > xmin and y1 < ymax and y1 > ymin and z1 < zmax and z1 > zmin and x2 < xmax and x2 > xmin and y2 < ymax and y2 > ymin and z2 < zmax and z2 > zmin):
                    T=False
                    continue
                else:

                    # 再来一遍
                    # 定义新个体
                    new_individual1 = ''
                    new_individual2 = ''
                    crossover_point = random.randint(1, len(population_cross[i][0]) - 1)  # 随机选择交叉点
                    for j in range(len(population_cross[i][0])):
                        if j < crossover_point:
                            new_individual1 += population_cross[i][0][j]
                            new_individual2 += population_cross[i + 1][0][j]
                        else:
                            new_individual1 += population_cross[i + 1][0][j]
                            new_individual2 += population_cross[i][0][j]
                    T=True
                    time_end1 = time.time()
                    sum_time = time_end1 - time_start1
                    if (sum_time / 60 > 10):
                        T = False
                        flag1 = 1  # 表明时间太久了，直接跳过这个图片
                    break

        # 将新个体存储在新种群列表中
        new_population.append([new_individual1])
        new_population.append([new_individual2])

    # 输出交叉后的种群大小和内容
    print('交叉后的种群大小：', len(new_population))
    # print('交叉后的种群内容：', new_population)
    return new_population,flag1

# 非均匀变异
def non_uniform_mutation(new_population, mutation_rate,xmin,xmax,ymin,ymax,zmin,zmax,generation,max_generations):
    new_multi = []
    for i in range(len(new_population)):
        new_individual = ''
        for j in range(len(new_population[i][0])):
            b = new_population[i][0]
            # 计算非均匀变异率
            non_uniform_rate = mutation_rate * (1 - generation / max_generations)
            c = random.random()
            if c < non_uniform_rate:
                if new_population[i][0][j] == '1':
                    new_individual += '0'
                else:
                    new_individual += '1'
            else:
                new_individual += new_population[i][0][j]
        T=True
        while(T):
            T=False
            for k in range(0, len(new_individual), 96):  # 81120的长度96乘以845
                binary_x = new_individual[k:k + 32]
                binary_y= new_individual[k + 32:k + 64]
                binary_z = new_individual[k + 64:k + 96]
                x = struct.unpack('>f', bytes.fromhex(hex(int(binary_x, 2))[2:].zfill(8)))[0]
                y= struct.unpack('>f', bytes.fromhex(hex(int(binary_y, 2))[2:].zfill(8)))[0]
                z = struct.unpack('>f', bytes.fromhex(hex(int(binary_z, 2))[2:].zfill(8)))[0]
                if (x < xmax and x> xmin and y < ymax and y > ymin and z < zmax and z > zmin ):
                    T = False
                    continue
                else:
                    new_individual = ''
                    for j in range(len(new_population[i][0])):
                        b = new_population[i][0]
                        # 计算非均匀变异率
                        non_uniform_rate = mutation_rate * (1 - generation / max_generations)
                        c = random.random()
                        if c < non_uniform_rate:
                            if new_population[i][0][j] == '1':
                                new_individual += '0'
                            else:
                                new_individual += '1'
                        else:
                            new_individual += new_population[i][0][j]
                    T=True
                    break

        new_multi.append([new_individual])
                # 替换原来的基因

    return new_multi
def main():
    logger = common_utils.create_logger()
    false_pred=0
    number_bin=0
    for txt_file in txt_files:
        # 检查上次断开后已经跑的图片，如果已经存在，就重新选择图片
        passPath = "/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/最优扰动点/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard"
        # 获取文件夹中所有文件的文件名列表
        passFile_names = os.listdir(passPath)
        flag = False
        for passFile in passFile_names:
            if (txt_file == passFile):
                flag = True
                break
        if (flag == True):
            continue
        file_number = txt_file.split(".")[0]
        box_path = os.path.join(folder_path1, txt_file)
        Pedestrian_points_path = os.path.join(folder_path2, txt_file)
        box = np.loadtxt(box_path)
        column_means = np.mean(box, axis=0)
        Pedestrian_points = np.loadtxt(Pedestrian_points_path)
        intensity1 = np.max(Pedestrian_points, axis=0)[3]
        Pedestrian_points3 = Pedestrian_points[:, 0:3]
        args, cfg = parse_config(file_number)

        logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=logger
        )
        logger.info(f'Total number of samples: \t{len(demo_dataset)}')

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        data_dict = demo_dataset[0]
        points = data_dict['points']
        points_xyz = points[:, :3]
        np.savetxt('points_xyz.txt', points_xyz)

        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            pred_dicts, _ = model.forward(data_dict)

        pred_boxes = pred_dicts[0]['pred_boxes']
        pred_labels = pred_dicts[0]['pred_labels']  # 预测结果的类别
        pred_scores = pred_dicts[0]['pred_scores']  # 预测结果的置信度

        # 看被预测的在哪个位
        min1 = 1000
        m1 = 0
        m2 = 0
        for boxes in pred_boxes:
            dx = column_means[0] - boxes[0]
            dy = column_means[1] - boxes[1]
            dz = column_means[2] - boxes[2]
            dist1 = math.sqrt(dx * dx + dy * dy + dz * dz)
            if (min1 > dist1):
                min1 = dist1
                # m2记录最小值的索引
                m2 = m1
            m1 = m1 + 1

        if (min1 > mindist):
            print("不存在符合要求的车")
            continue

        first_pred_boxes = pred_boxes[m2]
        first_labels1 = pred_labels[m2]
        first_score=pred_scores[m2]

        num = 3
        for i in range(len(box)):
            if (box[i][0] < first_pred_boxes[0]):
                box[i][0] = box[i][0] - num
            else:
                box[i][0] = box[i][0] + num
            if (box[i][1] < first_pred_boxes[1]):
                box[i][1] = box[i][1] - num
            else:
                box[i][1] = box[i][1] + num
            if (box[i][2] < first_pred_boxes[2]):
                box[i][2] = box[i][2] - num
            else:
                box[i][2] = box[i][2] + num
        xmin = min(box[:, 0])
        xmax = max(box[:, 0])
        ymin = min(box[:, 1])
        ymax = max(box[:, 1])
        zmin = min(box[:, 2])
        zmax = max(box[:, 2])
        # 计算六个顶点的倒角距离的平均值
        # maxAverage = chamfer_distance(box, Pedestrian_points)


        # pred_scores=pred_scores[0]


        perturbation_4lie = []

        for j in range(20):
            # 生成随机扰动
            perturbation = np.random.normal(0, 0.01, size=Pedestrian_points3.shape)  # 使用标准正态分布生成与点云数据形状相同的随机扰动
            perturbed_points = Pedestrian_points3 + perturbation

            # 增加多一列
            # 新的一列，值为5
            new_column = np.full((perturbed_points.shape[0], 1), intensity1)

            # 将新列和原始数组拼接起来
            perturbation_4lie_zhi5 = np.concatenate((perturbed_points, new_column), axis=1)
            # 在数组中随机选取shi行,扰动点的个数
            random_rows = np.random.choice(perturbation_4lie_zhi5.shape[0], size=10, replace=True)

            perturbation_4lie_zhi5 = perturbation_4lie_zhi5[random_rows]

            # 将扰动后的对抗样本和扰动信息添加到列表中
            perturbation_4lie.append(perturbation_4lie_zhi5)
            new_population = perturbation_4lie
        #     将得到的扰动点
        # 选择操作
        # 每一次的最大适应度值
        max_fitness = []
        all_fitness_values = []
        all_individual = []
        all_pred_score = []
        all_label = []
        one_time_scores=[]
        one_time_label=[]

        all_iteration = []
        fitness=[]
        for individual in new_population:
            pred_scores, pred_labels = attack_GA1(points, individual, args, cfg, logger, first_pred_boxes, file_number,model)
            f = fitness_function(pred_scores, pred_labels, Pedestrian_points, individual)
            f = f.cpu().numpy()
            fitness.append(f)
            one_time_scores.append(pred_scores)
            one_time_label.append(pred_labels)
        for i in range(max_iterations):
            print('第', i + 1, '次迭代开始')

            selected_population =roulette_selection(new_population,fitness)

            # 单点交叉操作
            flag2=-1
            print(txt_file)
            crossover_population,flag1 = one_point_crossover(selected_population, pc, xmin, xmax, ymin, ymax, zmin, zmax)
            if(flag1==1):
                flag2=1
                break

            # 非均匀变异
            print("变异")
            mutation_population = non_uniform_mutation(crossover_population, mutation_rate, xmin, xmax, ymin, ymax,
                                                       zmin,
                                                       zmax, i, max_iterations)

            # 将变异之后的二进制转为浮点数
            population_float = []
            for binary_str in range(len(mutation_population)):
                binary1 = mutation_population[binary_str][0]
                individual = []
                for k in range(0, len(binary1), 96):  # 81120的长度96乘以845
                    binary_x = binary1[k:k + 32]
                    binary_y = binary1[k + 32:k + 64]
                    binary_z = binary1[k + 64:k + 96]
                    x = struct.unpack('>f', bytes.fromhex(hex(int(binary_x, 2))[2:].zfill(8)))[0]
                    y = struct.unpack('>f', bytes.fromhex(hex(int(binary_y, 2))[2:].zfill(8)))[0]
                    z = struct.unpack('>f', bytes.fromhex(hex(int(binary_z, 2))[2:].zfill(8)))[0]
                    individual.append([x, y, z])
                population_float.append(individual)

            new_population1 = []

            for individual in population_float:
                # 创建新的一列
                individual = np.array(individual)
                new_column = np.full((individual.shape[0], 1), intensity1)
                # 将原数组和新的一列合并成一个新的数组
                new_individual = np.concatenate((individual, new_column), axis=1)
                new_population1.append(new_individual)

            # 评估适应度
            print("第", i + 1, "次攻击之后。。。。。。。。。。。。。。。。。。。。。。")
            for individual in new_population1:
                print("333333333333333333333333333333333333333333333333")
                pred_scores, pred_labels = attack_GA1(points, individual, args, cfg, logger, first_pred_boxes,file_number,model)
                f = fitness_function(pred_scores, pred_labels, Pedestrian_points, individual)
                f = f.cpu().numpy()
                fitness.append(f)  # 一共有20个置信度

                one_time_scores.append(pred_scores)
                one_time_label.append(pred_labels)


            # new_population= roulette_selection(np.concatenate((new_population, new_population1), axis=0), fitness_values)
            new_population,fitness,one_time_scores,max_individual,max_score,one_time_label,max_label = elitism_tournament_selection(np.concatenate((new_population, new_population1), axis=0),
                                                          fitness, num_elites, tournament_size,one_time_scores,one_time_label)
            # new_population,fitness=elitism_selection(np.concatenate((new_population, new_population1), axis=0),fitness_values, num_elites)
            max_fitness.append(max(fitness))
            # all_fitness_values.append(max_fitness)
            all_iteration.append(i)
            all_individual.append(max_individual)
            all_pred_score.append(max_score)
            all_label.append(max_label)

            print("第", i + 1, "次迭代结束")
        if(flag2==1):
            continue

        best_all_iteration = all_iteration[np.argmax(max_fitness)]
        best_individual = all_individual[np.argmax(max_fitness)]
        best_all_pred_score = all_pred_score[np.argmax(max_fitness)]
        best_all_label = all_label[np.argmax(max_fitness)]
        output_file=os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/输出控制台结果/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard", txt_file)
        number_bin += 1
        with open(output_file, "w") as file:
            # 保存默认的标准输出对象
            original_stdout = sys.stdout
            # 将标准输出重定向到文件
            sys.stdout = file
            # 执行多个print语句，输出将保存到文件中
            print("原来的置信度为：",first_score)
            print("原来的标签为：",first_labels1)
            print('最大适应度', max(max_fitness))
            print('第', best_all_iteration, '次取得适应度的最大值')
            print('对应的扰动点为：', best_individual)
            print('对应的置信度为：', best_all_pred_score)
            print('对应的标签为：', best_all_label)
            if (best_all_label != 2):
                false_pred+=1
                print("攻击成功")
                # 恢复标准输出到控制台
            time_end = time.time()  # 记录结束时间
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print("time_sum", time_sum)
            print("攻击成功率为:", false_pred / number_bin, false_pred, number_bin)
            sys.stdout = original_stdout

        print("输出已保存到文件:", output_file)
        best_individual_file=os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/最优扰动点/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard", txt_file)
        np.savetxt(best_individual_file, best_individual)
        max_fitness_file=os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/每一次迭代最大适应度值绘制/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard", txt_file)
        np.savetxt(max_fitness_file, max_fitness)
        # 绘制每一次得到的最大适应度的曲线图
        plt.figure()
        plt.plot(max_fitness, 'r', label='max_fitness')
        plt.ylabel('max_fitness')
        plt.xlabel('iter_num')
        plt.legend()
        plt.savefig(os.path.join("/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/每一次迭代最大适应度值绘制/pointrcnn/1000/10个扰动点/adv2_Pedestrian/Hard", f"{file_number}.jpg"))
        print(file_number)
        if(number_bin==50):
            break
        print("攻击完-------------",file_number)
    print("攻击成功率为:",false_pred/number_bin,false_pred,number_bin)

if __name__ == '__main__':
    main()
