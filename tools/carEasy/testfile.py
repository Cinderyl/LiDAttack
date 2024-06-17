import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
time_start = time.time()  # 记录开始时间
import argparse
import glob
import math
from pathlib import Path
import random
import struct
try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import tools.mayavi.mlab as mlab
    from tools.visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import matplotlib.pyplot as plt

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
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../cfgs/kitti_models/pointrcnn2.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/data0/benke/ldx/openpcdet37/OpenPCDet/data/kitti/training/velodyne/000010.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--per_data_path', type=str, default='/data0/benke/ldx/openpcdet37/OpenPCDet/tools/中间bin文件/000002.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/data0/benke/ldx/openpcdet37/OpenPCDet/output/cfgs/kitti_models/pointrcnn/default/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
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
    print(pred_boxes)
    pred_scores=pred_dicts[0]['pred_scores']
    print(pred_scores)
    #
    # # 定义Grad-CAM类的实例
    # grad_cam = GradCAM(model)
    #
    # # 准备输入点云数据，这里假设输入数据为点云矩阵points，大小为[N, 3]，其中N为点的数量
    # # points = torch.tensor(points_xyz, dtype=torch.float32).unsqueeze(0)
    #
    # # 计算Grad-CAM梯度权重
    # grad_cam_weights = grad_cam.calculate_gradients(data_dict)
    # print(grad_cam_weights)
    # 将梯度权重应用到输入点云数据上，得到可视化结果
    # visualized_points = grad_cam.apply_heatmap(points, grad_cam_weights)
    #
    # # 可视化原始点云数据和Grad-CAM结果
    # plt.scatter(points[:, 0], points[:, 1], c='blue', s=5, label='Original Point Cloud')
    # plt.scatter(visualized_points[:, 0], visualized_points[:, 1], c='red', s=5, label='Grad-CAM Visualization')
    # plt.legend()
    # plt.show()





if __name__ == '__main__':
    main()
