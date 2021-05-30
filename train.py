# Author: Jintao Huang
# Time: 2020-6-7


import torch
import torch.nn as nn
from models.efficientnet import efficientnet_b4, efficientnet_b2, efficientnet_b0
import os
from utils.tools import Trainer, Logger, Tester, Checker, Saver, LRScheduler, get_dataset_from_pickle, AccCounter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.display import resize_pad, random_perspective, augment_hsv, random_crop
from models.utils import label_smoothing_cross_entropy as _label_smoothing_cross_entropy, cosine_annealing_lr
import random


# # width_ratio, depth_ratio, resolution[% 32 may be not == 0], dropout_rate
# 'efficientnet_b0': (1.0, 1.0, 224, 0.2),
# 'efficientnet_b2': (1.1, 1.2, 260, 0.3),
# 'efficientnet_b4': (1.4, 1.8, 380, 0.4),
def label_smoothing_cross_entropy(pred, target):
    return _label_smoothing_cross_entropy(pred, target, smoothing)


efficientnet = efficientnet_b2
smoothing = 0.01  # loss_fn超参数
loss_fn = label_smoothing_cross_entropy
weight_decay = 1e-4
batch_size = 32
num_workers = 4
image_size = 260
epochs = (0, 50)
warm_up_steps = 200  # warm_up迭代次数(只在epoch==0时生效)
warm_up_bias = 0.05  # warm_up初始bias. 初始weight为0.
# 正式训练
min_lr, max_lr = 0.001, 0.01  # 余弦退火学习率
# ---
fill_value = 114  # 图像增强空白处填充
degrees = 0.5  # 旋转度数. 顺逆时针随机
translate = 0.1  # 移动比例.
scale = 0.05  # 缩放比例. 0.95 ~ 1.05
sheer = 0.5  # 斜切
perspective = 0  # 投影变换
random_crop_p = 0.6  # 随机裁剪概率
crop_scale_range = (0.5, 1)  # 裁剪参数
hsv = 0.015, 0.7, 0.4  # hsv色彩变换
# 测试
augment_test = True

comment = {
    "efficientnet": efficientnet.__name__,
    "smoothing": smoothing,
    "loss_fn": loss_fn.__name__,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "image_size": image_size,
    "num_workers": num_workers,
    "epochs": epochs,
    "warm_up_steps": warm_up_steps,
    "warm_up_bias": warm_up_bias,
    "min_lr": min_lr,
    "max_lr": max_lr,
    "fill_value": fill_value,
    "degrees": degrees, "translate": translate, "scale": scale, "sheer": sheer, "perspective": perspective,
    "random_crop_p": random_crop_p, "crop_scale_range": crop_scale_range,
    "hsv": hsv,
    "augment_test": augment_test,
}

# --------------------------------
dataset_dir = r'./dataset'  # 数据集所在文件夹
pkl_folder = 'pkl/'
train_pickle_fname = "images_targets_train.pkl"
val_pickle_fname = "images_targets_val.pkl"

labels = [
    "person", "car"
]

image_size = (image_size, image_size) if isinstance(image_size, (int, float)) else image_size


# --------------------------------


def linear_lr(steps, max_steps, lr_start, lr_end):
    if steps >= max_steps:
        return lr_end
    return lr_start + (lr_end - lr_start) / max_steps * steps


def lr_func(epoch, steps=None):
    if steps < warm_up_steps and epoch == 0:
        # bn_weight, weight, bias
        return [
            linear_lr(steps, warm_up_steps, 0, max_lr),
            linear_lr(steps, warm_up_steps, 0, max_lr),
            linear_lr(steps, warm_up_steps, warm_up_bias, max_lr)
        ]
    else:
        return cosine_annealing_lr(epoch, epochs[1] - 1, min_lr, max_lr)


def train_transform(image, target):
    """

    :param image: ndarray[H, W, C] BRG
    :param target:
    :return: ndarray[H, W, C] BRG"""

    image = resize_pad(image, image_size, False, 32, False, fill_value)[0]
    if random.random() <= random_crop_p:
        image = random_crop(image, crop_scale_range, fill_value)
    image = random_perspective(image, degrees, translate, scale, sheer, perspective, fill_value)
    image = augment_hsv(image, *hsv)
    return image, target


def test_transform(image, target):
    """由于测试集准确率过高，无法确定模型优略，所以使用加增强的测试

    :param image: ndarray[H, W, C] BRG
    :param target: 不作处理
    :return: ndarray[H, W, C] BRG"""
    image = resize_pad(image, image_size, False, 32, False, fill_value)[0]
    if augment_test:
        if random.random() <= random_crop_p:
            image = random_crop(image, crop_scale_range, fill_value)
        image = augment_hsv(image, hsv[0] / 3, hsv[1] / 3, hsv[2] / 3)
    return image, target


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = efficientnet(True, image_size=image_size, num_classes=len(labels))
    # 断点续训
    # load_params(model, "XXX.pth")
    # freeze_layers(model, ["conv_first", "layer1"])
    pg0, pg1, pg2 = [], [], []  # bn_weight, weight, bias
    for k, v in model.named_modules():
        # bias
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases. no decay
        # weight
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optim = torch.optim.SGD(pg0, 0, 0.9)  # bn_weight
    optim.add_param_group({'params': pg1, 'weight_decay': 1e-4})  # add pg1 with weight_decay
    optim.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    train_dataset = get_dataset_from_pickle(os.path.join(dataset_dir, pkl_folder, train_pickle_fname), train_transform)
    val_dataset = get_dataset_from_pickle(os.path.join(dataset_dir, pkl_folder, val_pickle_fname), test_transform)
    train_loader = DataLoader(train_dataset, batch_size, True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, False, pin_memory=True, num_workers=num_workers)
    acc_counter = AccCounter(labels)
    saver = Saver(model)
    save_dir = saver.save_dir
    print("配置: %s" % comment, flush=True)
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for k, v in comment.items():
            f.write("%s: %s\n" % (k, v))
    writer = SummaryWriter(logdir=save_dir)
    logger = Logger(50, writer)
    checker = Checker({"Test": Tester(model, val_loader, device, acc_counter, "all")},
                      saver, 1, 0, logger)
    lr_scheduler = LRScheduler(optim, lr_func)
    trainer = Trainer(model, train_loader, loss_fn, optim, device, lr_scheduler, logger, checker)
    trainer.train(epochs)
    writer.close()


if __name__ == "__main__":
    main()
