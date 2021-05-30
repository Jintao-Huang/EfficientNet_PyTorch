# author: Jintao Huang
# date: 2020-5-14
# This case applies to Debug to view the network architecture

from models.efficientnet import efficientnet_b0, std_preprocess, config_dict
import torch
from utils.display import resize_pad
import numpy as np
import cv2 as cv
import torch.nn as nn
from utils.utils import processing


def pred_transform(image, target):
    """

    :param image: ndarray[H, W, C] RGB
    :param target: None
    :return: ndarray[H, W, C] RGB 0-255, None"""
    image = resize_pad(image, image_size, False, 32, False, 114)[0]
    return image, target


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
efficientnet = efficientnet_b0
image_size = config_dict[efficientnet.__name__][2]

# read images
image_fname = "images/1.jpg"
x = cv.imread(image_fname, cv.IMREAD_COLOR)
x = processing(x, pred_transform)[0].to(device)[None] / 255
y_true = torch.randint(0, 2, (1,)).to(device)

model = efficientnet(pretrained=True, num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), 1e-3, 0.9)

for i in range(20):
    pred = model(x)
    loss = loss_fn(pred, y_true)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("loss: %f" % loss.item())
