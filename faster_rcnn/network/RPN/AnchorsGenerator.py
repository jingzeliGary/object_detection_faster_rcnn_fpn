'''
AnchorsGenerator:
基于每个 feature map, 在原图上生成 不同 sizes, aspect_ratios 的 anchors
1. 根据提供的sizes和aspect_ratios生成anchors模板---> cell_anchors
    (anchors模板都是以(0, 0)为中心的anchor)
2. 根据每个feature map size 和 对应原图上的步长， 计算 每个 feature map 对应 原始图像上的 所有anchors的坐标
    将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape 使用广播机制) *****
3. 遍历每张图片， 生成anchors
'''

import torch
import torch.nn as nn

class AnchorsGenerator(nn.Module):
    """
    anchors生成器
    基于每个 feature map, 在原图上生成 不同 sizes, aspect_ratios 的 anchors
    """
    def __init__(self, sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
        super(AnchorsGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None  # anchors 模板
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        Arguments:
            scales: sqrt(anchor_area)
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round 四舍五入

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        '''
       根据提供的sizes和aspect_ratios生成anchors模板---> cell_anchors
        anchors模板都是以(0, 0)为中心的anchor
       '''
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors

            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        '''
        计算 每个 feature map 对应 原始图像上的 所有anchors的坐标
        :param grid_sizes: feature map 的height和width, (h,w)
        :param strides: feature map 对应 原图 的 映射步距, (stride_h, stride_w)
        '''
        anchors = []
        cell_anchors = self.cell_anchors

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # feature map 对应 原图的 x, y坐标轴
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 生成坐标点位置
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape 使用广播机制) *****
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        '''
        Args:
            feature_sizes: 每个特征图的尺寸, List(Tensor(feature_h, feature_w))
            strides: 每个特征图对应输入图像的尺寸, List(Tuple(stride_h, stride_w))
        '''
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides) # List[Tensor(all_num_anchors, 4)]
        self._cache[key] = anchors
        return anchors  # List[Tensor(all_num_anchors, 4)]

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        '''
       Args:
           image_list: tensors: Tensor(batch_size, c, h, w) ; image_sizes: List(Tuple(resize_h, resize_w))
           feature_maps: List(Tensor(batch_size, c, h, w))
       '''
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        # List(Tuple(feature_h,feature_w))

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]  # [h,w]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # 计算 每个特征层 对应 输入图像 的 步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        # strides: List(Tuple(stride_h, stride_w))

        # 根据提供的sizes和aspect_ratios生成anchors模板 返会 self.cell_anchors
        self.set_cell_anchors(dtype, device)

        # 将 每层feature map 上的 anchors 模板 映射的 原图上
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = []  # List(List(Tensor(all_num_anchors, 4))) batch中一张图片的一个feature map的 对应原图的anchors 坐标
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors  # List(anchors_all_feature_maps, 4)