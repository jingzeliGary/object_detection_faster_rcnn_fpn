'''
RegionProposalNetwork:
RPN Head, AnchorGenerator,
基于 box regs和 生成的 anchors, 计算 proposals
Filter proposals:   获取 每个 feature map 上预测概率排前pre_nms_top_n的anchors索引值
                    防止建议框超出图像边缘,
                    筛除 小box框, 移除小score框， nms,
                    根据 scores 获取前 post_nms_top_n个目标
# training:
assign_targets_to_anchors: 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
encode: 基于每个anchors最匹配的gt 和 anchors 计算 real_box_regs
compute_loss: 分类损失: sigmoid损失, F.binary_cross_entropy_with_logits
             回归损失： smooth_L1
'''
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import network.det_utils as det_utils
import network.boxes as box_ops

class RegionProposalNetwork(nn.Module):
    """
    RPN: 生成 proposal
    anchor_generator: 在原图上生成不同尺寸的 anchors
    rpn_head : 3*3 滑动窗口 预测 目标概率和回归参数 pred_box_regs
    decode : pred_box_regs + anchors -> proposals
    filter_proposals: 过滤低质量, 重叠 的 proposal

    if self.training:
        计算每个anchors 匹配的 gt, 将 anchors 分为 正样本 和 负样本
        基于 anchors和其匹配的gt, 计算 real_box_regs
        基于 rpn_head输出的目标概率, 预测回归参数, anchor的label(是否正样本）, real_box_regs, 计算分类损失， 回归损失

    """
    def __init__(self, anchor_generator, head,
                 # training
                 fg_iou_thresh, bg_iou_thresh,  # 0.7, 0.3
                 batch_size_per_image, positive_fraction,  # 256, 0.5
                 # testing
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))  # RPN-weights=(1.0,1.0,1.0,1.0)

        # training
        # 正负样本匹配, 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
        # 返回 Tensor(nums_anchors)  -1 负样本， -2 丢弃样本， 0，1.. 正样本对应的gt box 索引
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True  # 补充正样本
        )
        # 正负样本采样
        # 返回  List(Tensor(nums_anchors))， # [1, 0. 1..] 1: 正样本， 0: 负样本/丢弃样本
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: List(Tensor(anchors_all_feature_maps, 4))
            targets: List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            # 计算anchors与真实bbox的iou信息
            match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
            # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
            matched_idxs = self.proposal_matcher(match_quality_matrix) \
                # Tensor(nums_anchors)  -1 负样本， -2 丢弃样本， 0，1.. 正样本对应的gt box 索引

            # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
            # 负样本和舍弃的样本都是负值，所以为了防止越界直接置为0
            # 因为后面是通过labels_per_image变量来记录正样本位置的，
            # 所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，
            # 反正计算目标边界框回归损失时只会用到正样本。
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            # 记录所有anchors匹配后的标签(正样本处标记为1，负样本处标记为0，丢弃样本处标记为-2)
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # 背景 = 0
            bg_indices = matched_idxs == -1
            labels_per_image[bg_indices] = 0.0

            # 丢弃样本 = -1
            inds_to_discard = matched_idxs == -2
            labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes
        # labels: List(Tensor(nums_anchors)), 0 负样本， -1 丢弃样本， 1 正样本
        # matched_gt_boxes: List(Tensor(nums_anchors_gt, 4))

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0  # 偏移量
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        获取 每个 feature map 上预测概率排前pre_nms_top_n的anchors索引值, 防止建议框超出图像边缘,
        筛除 小box框, 移除小score框， nms, 根据 scores 获取前 post_nms_top_n个目标
        Args:
            proposals: 预测的 box坐标,  Tensor(batch_size, anchors_all_feature_maps,4)
            objectness: 预测的目标概率, Tensor(batch_size * num_anchors_all_feature_maps, 1)
            image_shapes: batch中每张图片的 resize信息, List(Tuple(resize_h, resize_w))
            num_anchors_per_level: 每个预测特征层上预测anchors的数目, List(num_anchors * h * w)
        """
        num_images = proposals.shape[0]
        device = proposals.device

        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)  # Tensor(batch_size, num_anchors_all_feature_maps)

        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]  # List(Tensor(0,0,0..), Tensor(1,1,1..)]
        levels = torch.cat(levels, 0)  # Tensor(nums_feature_maps * num_anchors * h * w)

        levels = levels.reshape(1, -1).expand_as(objectness)  # Tensor(batch_size, num_anchors_all_feature_maps)

        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        # top_n_index : Tensor(batch_size, pre_nms_top_n * nums_feature)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息, 所属的feature_map信息
        objectness = objectness[batch_idx, top_n_idx]  # Tensor(batch_size, pre_nms_top_n)
        levels = levels[batch_idx, top_n_idx]  # Tensor(batch_size, pre_nms_top_n)
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]  # Tensor(batch_size, pre_nms_top_n, 4)

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            '''
            boxes, Tensor(pre_nms_top_n, 4)
            scores,  Tensor(pre_nms_top_n)
            lvl,  Tensor(pre_nms_top_n)
            img_shape: Tuple(resize_h, resize_w))
            '''
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 筛选小 box 框
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            keep = torch.where(scores > self.score_thresh)[0]   # torch.where --> tuple
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 为每层进行 NMS
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # 保留前 post_nms_top_n 个 boxes, scores
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        Arguments:
            objectness (Tensor)：预测的前景概率,  Tensor(batch_size * num_anchors_all_feature_maps, 1)
            pred_bbox_deltas (Tensor)：预测的bbox regression, Tensor(batch_size * num_anchors_all_feature_maps, 4)
            labels (List[Tensor])：List(Tensor(nums_anchors))
            regression_targets (List[Tensor])：真实的bbox regression, # List(Tensor(nums_anchor, 4)
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]

        # features是所有预测特征层组成的OrderedDict
        features = list(features.values())

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        objectness, pred_bbox_deltas = self.head(features)

        # 生成一个batch图像的所有anchors信息,list(tensor)元素个数等于batch_size
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # 计算每个预测特征层上的对应的anchors数量
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整内部tensor格式以及shape
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)

        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C,  H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    对 scores和 box_regs 每个预测特征层的预测信息的 tensor排列顺序以及shape进行调整 -> [N, -1, C]
    Args:
        box_cls: 每个 feature map 上的预测目标概率                      List(Tensor(batch_size, num_anchors, h,w))
        box_regression: 每个 feature map 上的预测目标bboxes regression参数   List(Tensor(batch_size, num_anchors * 4, h, w))

    """
    box_cls_flattened = []
    box_regression_flattened = []

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression
