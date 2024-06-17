import torch
import torch.nn as nn
import torch.optim as optim
from pcdet.models import build_detector

# 1. 定义 PointRCNN 模型
cfg_file = '../cfgs/kitti_models/pointrcnn2.yaml'
model = build_detector(cfg_file)
model.train()

# 2. 准备输入数据
# 假设你已经准备好了 DataLoader，并加载了点云数据和 bounding box 等
input_data = torch.tensor(...)  # 输入点云数据
bounding_box_data = torch.tensor(...)  # bounding box 数据

# 3. 前向传播
output = model(input_data, bounding_box_data)

# 4. 选择目标层
target_layer = model.conv1  # 假设选择 PointRCNN 模型的第一个卷积层为目标层

# 5. 反向传播
output.backward()

# 6. 访问目标层的梯度
target_layer_gradient = target_layer.grad
print("目标层的梯度：", target_layer_gradient)
