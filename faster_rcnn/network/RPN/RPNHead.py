'''
RPN Head:
通过滑动窗口计算预测目标概率与bbox regression参数
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class RPNHead(nn.Module):
    """
    RPHHead:
    通过滑动窗口计算预测目标概率与bbox regression参数
    基于 每个 feature map 进行 3*3卷积， 再分别进行 1*1 卷积 计算 类别分数 和 回归参数
    """
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # 初始化权重
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        # x: List(Tensor(batch_size, c, h, w))
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg  # logits: List(Tensor(batch_size, num_anchors, h,w))
                                 # bbox_reg: List(Tensor(batch_size, num_anchors * 4, h, w))