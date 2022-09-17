import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
from network.GeneralizedRCNNTransform.image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    """
    Normalization
    Resize
    batch_image: 填充打包成batch

    return:
    ImageList : tuple(tensors, image_sizes), tensors: Tensor, image_sizes: List(resized_h, resized_w)
    targets: List(Dict)
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息
        Args:
            image: 输入的图片, Tensor
            target: 输入图片的标注信息, List(Dict)
        """
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]

        image = _resize_image(image, float(self.min_size), float(self.max_size))

        if target is None:
            return image, target

        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def batch_images(self, images):
        # type: (List[Tensor], int) -> Tensor
        """
        将图像填充成相同的shape, 打包成一个batch返回
        Args:
            images: 输入的一批图片, List(Tensor)
        """

        # 分别计算一个batch中所有图片中的最大channel, height, width
        max_c, max_h, max_w = tuple(max(s) for s in zip(*[image.shape for image in images]))

        # [batch, channel, height, width]
        batch_shape = [len(images)] + [max_c, max_h, max_w]

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            image = self.normalize(image)   # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)   # 对图像和对应的bboxes缩放到指定范围
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 将images打包成一个batch
        image_sizes_list = []

        for image_size in image_sizes:
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))    # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))    # 获取高宽中的最大值
    scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)







