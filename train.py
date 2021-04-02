# Author: Jintao Huang
# Time: 2020-6-7

import torch
from models.efficientnet import efficientnet_b2, preprocess
import os
from utils.detection import Trainer, Logger, Tester, Checker, Saver, LRScheduler, get_dataset_from_pickle, AccCounter
from tensorboardX import SummaryWriter
from models.utils import freeze_layers

batch_size = 64
comment = "-b2,wd=4e-5,bs=64,lr=0.05"

# --------------------------------
dataset_dir = r'../fer2013'
train_dir = os.path.join(dataset_dir, "Training")
test_dir = os.path.join(dataset_dir, "PublicTest")
pkl_folder = '../pkl/'
train_pickle_fname = "images_targets_train.pkl"
test_pickle_fname = "images_targets_test.pkl"

labels = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"
]


# --------------------------------
def lr_func(epoch):
    if 0 <= epoch < 1:
        return 1e-3
    elif 1 <= epoch < 3:
        return 0.01
    elif 3 <= epoch < 32:
        return 0.05
    elif 32 <= epoch < 37:
        return 0.02
    elif 37 <= epoch < 40:
        return 5e-3


def transform(image):
    return preprocess([image], 96)[0]


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = efficientnet_b2(True, image_size=96, num_classes=len(labels))
    # freeze_layers(model, ["conv_first", "layer1"])
    optim = torch.optim.SGD(model.parameters(), 0, 0.9, weight_decay=4e-5)
    train_dataset = get_dataset_from_pickle(os.path.join(train_dir, pkl_folder, train_pickle_fname), transform)
    test_dataset = get_dataset_from_pickle(os.path.join(test_dir, pkl_folder, test_pickle_fname), transform)
    acc_counter = AccCounter(labels)
    writer = SummaryWriter(comment=comment)
    logger = Logger(50, writer)
    checker = Checker(Tester(model, train_dataset, batch_size, device, acc_counter, 1000),
                      Tester(model, test_dataset, batch_size, device, acc_counter, 4000),
                      Saver(model), logger, 4, 0)
    lr_scheduler = LRScheduler(optim, lr_func)
    trainer = Trainer(model, optim, train_dataset, batch_size, device, lr_scheduler, logger, checker)
    print("配置: %s" % comment, flush=True)
    trainer.train((0, 40))
    writer.close()


if __name__ == "__main__":
    main()
