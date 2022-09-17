from typing import List, Tuple
from torch import Tensor


class ImageList(object):
    """
    ImageList : tuple( tensors, image_sizes) , tensors: Tensor(batch_size, c, h, w), image_sizes: List(Tensor(resized_h, resized_w))
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

