import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    #  计算梯度
    def calculate_gradients(self,data_dict):
        # data_dict.requires_grad = True

        # with torch.no_grad():
        pred_dicts, _ = self.model.forward(data_dict)

        # 得到对每个类别的预测概率
        output =pred_dicts[0]['pred_scores']
        # 预测概率最大的索引
        target_class = torch.argmax(output)

        # # 将模型的参数梯度清零。这是为了避免之前计算的梯度对当前计算的梯度产生影响。
        self.model.zero_grad()
        # # 计算目标类别的预测概率关于输入数据的梯度
        output[target_class].requires_grad_(True).backward()
        #
        gradients = data_dict.grad
        # 计算目标类别的预测概率关于输入数据的梯度
        # input_data.requires_grad_(True)
        # gradients = torch.autograd.grad(output[target_class], input_data, retain_graph=True)[0]
        return gradients

    def apply_heatmap(self, input_data, gradients):
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        heatmap = F.relu(torch.sum(weights * input_data, dim=1))

        return heatmap

# 使用示例
# 1. 加载PointRCNN模型并定义input_data
# 2. 创建GradCAM实例
# 3. 计算梯度权重
# 4. 将梯度权重应用到input_data上，得到可视化结果
