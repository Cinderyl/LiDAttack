import torch  # 命令行是逐行立即执行的
content = torch.load('/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/output/cfgs/custom_models/pointrcnn/default/ckpt/checkpoint_epoch_51.pth')
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
print(content['optimizer_state'])
