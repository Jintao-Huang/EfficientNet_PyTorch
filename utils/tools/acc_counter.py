# Author: Jintao Huang
# Time: 2020-6-6
import torch
import numpy as np
from collections import OrderedDict


class AccCounter:
    def __init__(self, labels):
        """

        :param labels: List[str]
        """
        self.labels = labels
        self.acc_sum_num = None

    def init(self):
        self.acc_sum_num = np.zeros((len(self.labels), 2))  # sum, num

    def add(self, pred, target):
        """

        :param pred: Tensor[N, In]
        :param target: Tensor[N]
        :return: None
        """
        pred = torch.argmax(pred, dim=1)
        for p, t in zip(pred, target):
            p, t = p.item(), t.item()
            self.acc_sum_num[t][1] += 1
            self.acc_sum_num[t][0] += (p == t)

    def get_acc_dict(self):
        acc_dict = OrderedDict()
        for label, acc_sum in zip(self.labels, self.acc_sum_num):
            acc_dict[label] = (acc_sum[0] / acc_sum[1]) if acc_sum[1] else 0.
        total = np.sum(self.acc_sum_num[:, 0]) / np.sum(self.acc_sum_num[:, 1])
        acc_dict["mean_acc"] = sum(acc_dict.values()) / len(acc_dict)
        acc_dict["total_acc"] = total
        return acc_dict

    @staticmethod
    def print_acc(acc_dict):
        print("ACC: ")
        for label, acc in acc_dict.items():
            print("  %s: %.4f%%" % (label, acc * 100))
        print("", end="", flush=True)
