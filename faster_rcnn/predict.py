'''
基于训练好的模型和权重， 预测
'''

import os
import time
import json

import torch


import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from network.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network.faster_rcnn_framework import FasterRCNN
from draw_box_utils import draw_objs


def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 加载模型
    model = create_model(num_classes=3)

    # 加载训练好的权重
    weights_path = "./save_weights/resNetFpn-model-5.pth"
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    # 读取类别索引
    label_json_path = './pascal_voc_classes.json'
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # 加载测试图片
    original_img = Image.open("./test_image/3.jpg")

    # PIL--> Tensor
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)

    img = torch.unsqueeze(img, dim=0)  # Tensor(1, c, h, w)

    model.eval()  # 进入验证模式
    with torch.no_grad():

        t_start = time.time()
        predictions = model(img.to(device))[0]
        t_end = time.time()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
