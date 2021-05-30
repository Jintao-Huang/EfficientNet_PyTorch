# Author: Jintao Huang
# Time: 2020-6-7
import time


class Logger:
    def __init__(self, print_steps, writer=None):
        """Notice: 需要显式的关闭writer. `writer.close()`"""
        self.writer = writer
        self.print_steps = print_steps
        self.steps_each_epoch = None
        # ----------------
        self.epoch = None
        self.lr = None
        self.steps = None
        self.loss = None
        self.epoch_start_time = None
        self.mini_start_time = None

    def new_epoch(self, epoch, steps_each_epoch):
        self.epoch = epoch
        self.steps_each_epoch = steps_each_epoch
        self.steps = 0
        self.loss = []
        self.epoch_start_time = time.time()
        self.mini_start_time = time.time()

    def step(self, loss, lr):
        self.steps += 1
        self.lr = lr
        self.loss.append(loss)
        if self.steps % self.print_steps == 0 or self.steps == self.steps_each_epoch:
            self._print_mes(last=self.steps == self.steps_each_epoch)
        if self.writer:
            self.log_mes({"loss/loss": loss, **{"lr/lr%d" % i: _lr for i, _lr in enumerate(lr)}})

    def log_mes(self, logs):
        for key, value in logs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in ("total_acc", "mean_acc"):
                        self.writer.add_scalar("%s_all/%s" % (key, k), v,
                                               self.epoch * self.steps_each_epoch + self.steps)
                    else:
                        self.writer.add_scalar("%s/%s" % (key, k), v, self.epoch * self.steps_each_epoch + self.steps)
            else:
                self.writer.add_scalar(key, value, self.epoch * self.steps_each_epoch + self.steps)

    def _print_mes(self, last=False):
        loss_mean = sum(self.loss) / len(self.loss)
        if last:
            time_ = time.time() - self.epoch_start_time
            print("Total ", end="")
        else:
            time_ = time.time() - self.mini_start_time
        print("Train| Epoch: %d[%d/%d (%.2f%%)]| Loss: %f| Time: %.4f| LR: %s" %
              (self.epoch, self.steps, self.steps_each_epoch, self.steps / self.steps_each_epoch * 100,
               loss_mean, time_, ",".join(["%.4g" % _lr for _lr in self.lr])), flush=True)
        self.mini_start_time = time.time()
