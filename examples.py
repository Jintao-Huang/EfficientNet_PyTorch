# author: Jintao Huang
# date: 2020-5-14

from models.efficientnet import efficientnet_b0
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

x = torch.rand(1, 3, 600, 600).to(device)
y_true = torch.randint(1000, (1,)).to(device)

model = efficientnet_b0(pretrained=True).to(device)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), 5e-4)

for i in range(20):
    pred = model(x)
    loss = loss_func(pred, y_true)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("loss: %f" % loss.item())
