import torch
import cv2
import numpy as np
from pcdet.models import build_detector
from pcdet.datasets import build_dataloader
from pcdet.utils import get_root_logger
from torch.nn import functional as F
import os
from pcdet.config import config

class GradCAM:
    def __init__(self, config_file, checkpoint_file, dataset_path):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.dataloader = self._build_dataloader()

    def _build_model(self):
        cfg = Config.fromfile(self.config_file)
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = torch.load(self.checkpoint_file, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _build_dataloader(self):
        cfg = Config.fromfile(self.config_file)
        dataset_cfg = cfg.data.test
        dataset_cfg.data_root = self.dataset_path
        dataset = build_dataset(dataset_cfg)
        dataloader_cfg = cfg.data.test_dataloader
        dataloader_cfg.dataset = dataset
        dataloader = build_dataloader(dataloader_cfg, dist=False, shuffle=False)
        return dataloader

    def _get_features(self, x):
        features = []
        for name, module in self.model.named_modules():
            x = module(x)
            if name == "backbone.conv1":
                features.append(x)
            elif name == "backbone.layer4":
                features.append(x)
                break
        return features

    def _get_gradcam(self, x, class_idx):
        features = self._get_features(x)
        output = self.model(x)
        output = output[0]
        output = F.softmax(output, dim=1)
        output = output[:, class_idx]
        self.model.zero_grad()
        output.backward(retain_graph=True)
        grads = []
        for name, module in self.model.named_modules():
            if name == "backbone.conv1":
                grad = module.weight.grad
                grad = F.adaptive_avg_pool2d(grad, (1, 1))
                grad = grad.squeeze()
                grads.append(grad)
            elif name == "backbone.layer4":
                grad = module.conv2.weight.grad
                grad = F.adaptive_avg_pool2d(grad, (1, 1))
                grad = grad.squeeze()
                grads.append(grad)
                break
        weights = torch.stack(grads).sum(dim=0)
        weights = F.relu(weights)
        weights /= torch.sum(weights)
        cam = torch.zeros(features[-1].shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * features[-1][0, i, :, :]
        cam = F.relu(cam)
        cam /= torch.max(cam)
        return cam.detach().cpu().numpy()

    def generate_gradcam(self, save_path):
        logger = get_root_logger()
        for i, data in enumerate(self.dataloader):
            logger.info(f"Processing image {i+1}/{len(self.dataloader)}")
            img = data["img"][0].to(self.device)
            img_path = data["img_metas"][0]["filename"]
            img_name = img_path.split("/")[-1]
            class_idx = 0  # we will visualize the first class
            cam = self._get_gradcam(img, class_idx)
            img = img.detach().cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            cam = cv2.resize(cam, img.shape[:2])
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img) / 255
            cam = cam / np.max(cam)
            cv2.imwrite(os.path.join(save_path, img_name), np.uint8(255 * cam))

config_file = "configs/kitti_models/pv_rcnn/pv_rcnn_4cls_kitti.py"
checkpoint_file = "checkpoints/pv_rcnn_4cls_kitti.pth"
dataset_path = "/path/to/kitti/dataset"
save_path = "/path/to/save/gradcam/results"
gradcam = GradCAM(config_file, checkpoint_file, dataset_path)
gradcam.generate_gradcam(save_path)
