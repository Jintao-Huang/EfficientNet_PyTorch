# Author: Jintao Huang
# Time: 2020-6-7
from torch.utils.data import DataLoader
import torch
import math
from ..detection.utils import to


class Tester:
    def __init__(self, model, test_dataset, batch_size, device, acc_counter, test_samples=1000):
        self.model = model.to(device)
        self.test_loader = DataLoader(test_dataset, batch_size, True, pin_memory=True)
        iter(self.test_loader).__next__()
        self.device = device
        self.num_samples = len(test_dataset)
        self.batch_size = batch_size
        self.acc_counter = acc_counter
        self.test_step = math.ceil(test_samples / batch_size)

    def test(self, total=False):
        self.model.eval()
        with torch.no_grad():
            for i, (x, target) in enumerate(self.test_loader):
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
        self.acc_counter.print_ap(acc_dict)
