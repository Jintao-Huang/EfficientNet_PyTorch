# author: Jintao Huang
# date: 2020-5-14

import json
from models.efficientnet import efficientnet_b0, preprocess
import torch
from PIL import Image

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')

# read images
image_fname = "images/1.jpg"
with Image.open(image_fname) as x:
    x = preprocess([x], 224).to(device)

# read labels
with open('imagenet/labels_list.txt') as f:
    labels_map = json.load(f)

# pred
model = efficientnet_b0(pretrained=True).to(device)
model.eval()
with torch.no_grad():
    pred = torch.softmax(model(x), dim=1)
values, indices = torch.topk(pred, k=5)
print("Image Pred: %s" % image_fname)
print("-------------------------------------")
for value, idx in zip(values[0], indices[0]):
    value, idx = value.item(), idx.item()
    print("%-75s%.2f%%" % (labels_map[idx], value * 100))
