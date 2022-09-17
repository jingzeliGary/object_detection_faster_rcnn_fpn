
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.backbone.feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        # 主路
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # 支路
        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_structure, class_nums=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channel = 64
        self.layer1 = self.make_layer(block, block_structure[0], 64, stride=1)
        self.layer2 = self.make_layer(block, block_structure[1], 128, stride=2)
        self.layer3 = self.make_layer(block, block_structure[2], 256, stride=2)
        self.layer4 = self.make_layer(block, block_structure[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 （1，1）
        self.fc = nn.Linear(512 * block.expansion, class_nums)

        # 初始化参数 kaiming_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        '''
        # Xavier 均匀分布
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        # Xavier 正态分布
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

        # kaiming 均匀分布
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        # model: fan_in 正向传播，方差一致; fan_out 反向传播, 方差一致

        # 正态分布
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1)

        # 常量 , 一般是给网络中bias进行初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.constant_(m.bias, val=0)

        '''

    def make_layer(self, block, block_num, conv_num, stride):
        strides = [stride] + [1] * (block_num - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, conv_num, stride))
            self.in_channel = conv_num * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top is True:
            x = self.avgpool(x)  # [64, 512*block.expension, 1, 1]
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet50_fpn_backbone(pretrain_path=None,
                          returned_layers=None,
                          extra_blocks=None):
    """
    搭建resnet50_fpn——backbone
    Args:
        pretrain_path: resnet50的预训练权重，如果不使用就默认为空
        returned_layers: 指定哪些层的输出需要返回
        extra_blocks: 在输出的特征层基础上额外添加的层结构

    Returns: OrderDict('layer': feature_map)

    """
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3],include_top=False,)

    # 预训练权重
    if pretrain_path is not None:
        resnet50.load_state_dict(torch.load(pretrain_path), strict=False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # 返回的特征层个数肯定大于0小于5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet50.in_channel // 8  # 256
    # 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet50, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
