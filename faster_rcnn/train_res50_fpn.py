import os

import torch
import transforms
from dataset.my_dataset import VOCDataSet
from network.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network.faster_rcnn_framework import FasterRCNN
from network.ROI.FastRCNNPredictor import FastRCNNPredictor
from train_utils.train_eval_utils import train_one_epoch, evaluate
from train_utils.train_eval_utils import create_lr_scheduler


def create_model(num_classes, load_pretrain_weights=True):
    backbone = resnet50_fpn_backbone(pretrain_path="./network/backbone/pre_train_weights/resnet50.pth")
    model = FasterRCNN(backbone=backbone, num_classes=91)

    if load_pretrain_weights:
        # 载入预训练模型权重
        weights_dict = torch.load("./network/backbone/pre_train_weights/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results.txt"

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = './data'
    # 加载训练集
    train_dataset = VOCDataSet(VOC_root, data_transform["train"], "train.txt")

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    # 加载测试集
    val_dataset = VOCDataSet(VOC_root, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # 加载模型
    model = create_model(num_classes=3)  # nums_class + 1(bg)

    # 训练
    epochs = 6
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # 学习率预热
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), epochs, warmup=True)

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(epochs):
        # 训练一个epoch，每 2个batch 打印一次参数变化
        mean_loss, lr = train_one_epoch(train_data_loader, model, optimizer,
                                        device=device, epoch=epoch, lr_scheduler=lr_scheduler,
                                        print_freq=2)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 在测试集上评估
        coco_info = evaluate(model, val_data_set_loader, device=device)

        # 写入txt 文本
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # 保存 weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}

        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    main()
