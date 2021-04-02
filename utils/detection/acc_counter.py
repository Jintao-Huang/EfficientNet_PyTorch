# Author: Jintao Huang
# Time: 2020-6-6
import torch


class AccCounter:
    def __init__(self, labels):
        """

        :param labels: List[str]
        """
        self.labels = labels
        self.acc_sum_list = None

    def init(self):
        self.acc_sum_list = [[0., 0] for _ in range(len(self.labels))]

    def add(self, pred, target):
        """

        :param pred: Tensor[N, In]
        :param target: Tensor[N]
        :return: None
        """
        pred = torch.argmax(pred, dim=1)
        for p, t in zip(pred, target):
            p, t = p.item(), t.item()
            self.acc_sum_list[t][1] += 1
            self.acc_sum_list[t][0] += (p == t)

    def get_acc_dict(self):
        acc_dict = {}
        for label, acc_sum in zip(self.labels, self.acc_sum_list):
            acc_dict[label] = acc_sum[0] / acc_sum[1]
        return acc_dict

    @staticmethod
    def print_acc(acc_dict):
        mean_acc = sum(acc_dict.values()) / len(acc_dict)
        print("mean_ACC: %.4f%%" % mean_acc * 100)
        print("ACC: ")
        for label, acc in acc_dict.items():
            print("  %s: %.4f%%" % (label, acc * 100))
        print("", end="", flush=True)
