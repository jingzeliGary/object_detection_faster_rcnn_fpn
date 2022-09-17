import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from network.GeneralizedRCNNTransform.transform import GeneralizedRCNNTransform
from network.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network.RPN.AnchorsGenerator import AnchorsGenerator
from network.RPN.RPNHead import RPNHead
from network.RPN.RegionProposalNetwork import RegionProposalNetwork
from network.ROI.Two_MLP_Head import TwoMLPHead
from network.ROI.FastRCNNPredictor import FastRCNNPredictor
from network.ROI.roi_head import RoIHeads

class FasterRCNNBase(nn.Module):
    """
    Arguments:
        transform
        backbone
        rpn
        roi_heads
    """
    def __init__(self, transform, backbone, rpn, roi_heads):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        :param images: 原图, List(Tensor(c,h,w))
        :param targets: List(Dict)
        """
        # 记录原图大小
        original_image_sizes = [img.shape[-2:] for img in images]
        # 对图像进行预处理  images: tensors, image_sizes
        images, targets = self.transform(images, targets)
        # 将图像输入backbone得到特征图
        features = self.backbone(images.tensors)  # Orderdict('layer', Tensor(batch_size, c, h, w))

        # 将特征层以及标注target信息传入rpn中
        proposals, proposal_losses = self.rpn(images, features, targets)
        # proposals, List(post_nms_top_n, 4))
        # proposal_losses, Dict

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        else:
            return detections


class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,      # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):


        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        # ROI
        #  Multi-scale RoIAlign pooling, 要求的数据格式是 OrderDict
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        # transform (对数据进行标准化，缩放，打包成batch等处理部分)
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean=image_mean,
                                             image_std=image_std)

        super(FasterRCNN, self).__init__(transform, backbone, rpn, roi_heads)
