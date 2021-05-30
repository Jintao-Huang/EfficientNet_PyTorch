class LRScheduler:
    def __init__(self, optim, lr_func):
        self.optim = optim
        self.lr_func = lr_func

    def step(self, epoch, steps):
        lr = self.lr_func(epoch, steps)
        for i, pg in enumerate(self.optim.param_groups):
            pg['lr'] = lr if isinstance(lr, (float, int)) else lr[i]

    def get_lr(self):
        lr = []
        for pg in self.optim.param_groups:
            lr.append(pg['lr'])
        return lr
