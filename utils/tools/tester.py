# Author: Jintao Huang
# Time: 2020-6-7
import torch
import math
from ..tools.utils import to


class Tester:
    def __init__(self, model, test_loader, device, acc_counter, test_samples="all"):
        """

        :param test_samples: int or str
        """
        if test_samples.lower() == "all":
            test_samples = 0x7fffffff
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.acc_counter = acc_counter
        self.batch_size = test_loader.batch_size
        self.num_samples = len(test_loader) * self.batch_size
        self.test_step = math.ceil(test_samples / self.batch_size)

    def test(self, total=False):
        self.model.eval()
        self.acc_counter.init()
        with torch.no_grad():
            for i, (x, target) in enumerate(self.test_loader):
                x = x / 255
                x, target = to(x, target, self.device)
                pred = self.model(x)
                self.acc_counter.add(pred, target)
                if not total and i + 1 == self.test_step:
                    break
            acc_dict = self.acc_counter.get_acc_dict()
            self._print_mes(i + 1, acc_dict)
        self.acc_counter.init()  # clear memory
        return acc_dict

    def _print_mes(self, steps, acc_dict):
        test_num_samples = min(steps * self.batch_size, self.num_samples)
        print("Test | Samples: %d/%d (%.2f%%)" %
              (test_num_samples, self.num_samples,
               test_num_samples / self.num_samples * 100))
        self.acc_counter.print_acc(acc_dict)
