# author: Jintao Huang
# date: 2020-5-14

import json
from models.efficientnet import efficientnet_b0
import torch
import torchvision.transforms.transforms as transforms
from PIL import Image

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')

# read images
image_fname = "images/1.jpg"
with Image.open(image_fname) as x:
    x = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(x)[None].to(device)

# read labels
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

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

