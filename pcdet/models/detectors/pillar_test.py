from .detector3d_template import Detector3DTemplate

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            # 设置requires_grad为True，以便跟踪操作
            cur_module.requires_grad = True
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        # 获取dense_head的损失
        loss_rpn, tb_dict = self.dense_head.get_loss()

        # 清零梯度，防止累积
        self.zero_grad()

        # 反向传播
        loss_rpn.backward()

        # 获取模型参数的梯度
        model_gradients = {}
        for name, param in self.named_parameters():
            model_gradients[name] = param.grad

        # 在这里你可以使用model_gradients进行进一步的处理或打印
        print("Model Gradients:", model_gradients)

        # 将损失和其他信息添加到tb_dict中
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
