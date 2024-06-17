import open3d as o3d
import torch
import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.kitti.kitti_utils import calib_to_matricies as Calibration
import argparse
from pathlib import Path

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

    def forward(self, x):
        self.feature_maps = []
        self.gradient = None

        def hook_feature(module, input, output):
            self.feature_maps.append(output)

        def hook_gradient(grad):
            self.gradient = grad

        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_layer = module
                break

        handle_feature = target_layer.register_forward_hook(hook_feature)
        handle_gradient = target_layer.register_backward_hook(hook_gradient)

        with torch.enable_grad():
            output = self.model(x)
            class_idx = torch.argmax(output[0]['pred_scores'])
            one_hot = torch.zeros_like(output[0]['pred_scores'])
            one_hot[0][class_idx] = 1
            output[0]['pred_scores'].backward(gradient=one_hot, retain_graph=True)

        handle_feature.remove()
        handle_gradient.remove()

        return self.feature_maps[-1], self.gradient

    def generate_cam(self, x, class_idx):
        feature_map, gradient = self.forward(x)
        weights = torch.mean(gradient, axis=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * feature_map, axis=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.cpu().numpy()
        cam = cam.reshape(-1)
        return cam
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./cfgs/kitti_models/pointrcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str,
                        default='/data0/benke/ldx/openpcdet37/OpenPCDet/data/kitti',
                        help='specify the point cloud data file or directory')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)# 加载配置文件
    return args, cfg
args, cfg = parse_config()
# 加载点云数据
point_cloud = o3d.io.read_point_cloud('/data0/benke/ldx/openpcdet37/OpenPCDet/data/kitti/training/velodyne/000000.bin')
points = np.asarray(point_cloud.points)
points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # 添加齐次坐标
logger = common_utils.create_logger()
logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
# 构建数据集
dataset = KittiDataset(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    root_path=Path(args.data_path),
    training=False,
    logger=logger
)
data_dict = {
    'points': points,
    'frame_id': 0,
    'calib': Calibration('/data0/benke/ldx/openpcdet37/OpenPCDet/data/kitti/training/calib/000000.txt'),
    'image_shape': (cfg.DATA_CONFIG.IMAGE_HEIGHT, cfg.DATA_CONFIG.IMAGE_WIDTH)
}
data_dict = dataset.prepare_data(data_dict)

# 构建数据加载器
dataloader = build_dataloader(
    dataset,
    batch_size=1,
    dist=False,
    workers=0,
    logger=None
)

# 构建模型
model = build_network(cfg.MODEL, train_cfg=None, test_cfg=cfg.TEST_CFG)
model.load_params_from_file('../output/cfgs/kitti_models/pv_rcnn/default/ckpt/checkpoint_epoch_50.pth')
model.cuda()
model.eval()

# 计算模型输出
data = next(iter(dataloader))
data = common_utils.to_device(data, device='cuda')
with torch.no_grad():
    pred_dicts, _ = model.forward(data)

# 计算Grad-CAM
gradcam = GradCAM(model, target_layer='backbone.sa3.conv3')
class_idx = np.argmax(pred_dicts[0]['pred_scores'].cpu().numpy())
gradcam_map = gradcam.generate_cam(data['points'], class_idx)

# 将Grad-CAM叠加到点云上
gradcam_map = gradcam_map.reshape(-1)
gradcam_map = roiaware_pool3d_utils.average_pool(gradcam_map, data_dict['points'], data_dict['voxels'], data_dict['voxel_centers'], cfg.MODEL.DETECTION_HEAD.POOL_EXTRA_WIDTH)
gradcam_map = gradcam_map / np.max(gradcam_map)
gradcam_map = np.clip(gradcam_map, 0, 1)
colors = np.zeros_like(points)
colors[:, 0] = gradcam_map
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.io.write_point_cloud('/data0/benke/ldx/openpcdet37/OpenPCDet/data/gradcam.pcd', pcd)
