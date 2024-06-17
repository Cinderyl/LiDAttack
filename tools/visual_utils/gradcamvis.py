import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import load_data_to_gpu
from pcdet.models.backbones_3d import PointNet2MSG
from pcdet.models.detectors import PointRCNN
from GradCAM.gradcam import GradCam as GradCAM
from pcdet.config import cfg,cfg_from_yaml_file 
import matplotlib.pyplot as plt

#下载训练模型
backbone=PointNet2MSG(input_channels=4,use_xyz=True,cfg=None,model_cfg='./cfgs/kitti_models/pointrcnn.yaml')
detector=PointRCNN(
    num_classes=3,
    use_xyz=True,
    mode='train',
    use_reflective_loss=False,
    backbone=backbone,
    cfg=None

)
model = detector.cuda()
model.load_params_from_file('checkpoint_epoch_51.pth')
#定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
# Load the input point cloud data and apply the transform
dataset = KittiDataset(root_path='/data0/benke/ldx/openpcdet37/OpenPCDet/data/kitti',dataset='training')
data = dataset[0]
input_data = torch.from_numpy(data['points']).unsqueeze_(0).cuda()
inputdata = Variable(transform(input_data)).unsqueeze_(0)

# Define the target class
target_class=1

# Instantiate the Grad-CAM object
grad_cam =GradCAM(model=model)
# Generate the heatmap
heatmap =grad_cam(input_data,target_class)
# Visualize the heatmap
heatmap = heatmap.detach().cpu().numpy()[0]
plt.imshow(heatmap)
plt.show()