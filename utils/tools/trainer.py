# Author: Jintao Huang
# Time: 2020-6-6

import torch
from ..tools.utils import to


class Trainer:
    def __init__(self, model, train_loader, loss_fn, optim, device,
                 lr_scheduler=None, logger=None, checker=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optim = optim
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        assert checker
        self.checker = checker
        self.steps_each_epoch = len(self.train_loader)

    def train(self, epoch_range):
        for epoch in range(*epoch_range):
            self.model.train()
            self.logger.new_epoch(epoch, self.steps_each_epoch)
            for i, (x, target) in enumerate(self.train_loader):
                try:
                    self.lr_scheduler.step(epoch, epoch * self.steps_each_epoch + i) \
                        if self.lr_scheduler is not None else None
                    x = x / 255
                    x, target = to(x, target, self.device)
                    pred = self.model(x)
                    # loss = F.cross_entropy(pred, target)
                    loss = self.loss_fn(pred, target)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.logger.step(loss.item(), self.lr_scheduler.get_lr())
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    self.checker.saver.save("tmp_epoch%d_step%d" % (epoch, i + 1))
                    raise e

            if self.checker:
                self.checker.step(epoch, last=(epoch == epoch_range[1] - 1))
