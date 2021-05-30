# Author: Jintao Huang
# Time: 2020-6-6

import torch.utils.data as tud
import torch
from ..utils import load_from_pickle, processing
import cv2 as cv
import numpy as np


def get_dataset_from_pickle(pkl_path, transforms=None):
    img_path_list, target_list = load_from_pickle(pkl_path)
    return MyDataset(img_path_list, target_list, transforms)


class MyDataset(tud.Dataset):
    def __init__(self, img_path_list, target_list, transform=None):
        """

        :param img_path_list: List[str]
        :param target_list: List[int]
        :param transform: [](image: ndarray[H, W, C]BRG, target) -> image: ndarray[H, W, C]BRG, target
            默认(self._default_trans_func)
        """
        assert len(img_path_list) == len(target_list)
        self.img_path_list = img_path_list
        self.target_list = target_list
        self.transform = transform

    def __getitem__(self, idx):
        """

        :param idx:
        :return: Tensor[C, H, W] RGB
        """
        img_path = self.img_path_list[idx]
        target = self.target_list[idx]

        if isinstance(idx, slice):
            return self.__class__(img_path, target, self.transform)
        else:
            x = cv.imread(img_path)
            x, target = processing(x, target, self.transform)
            return x, target

    def __len__(self):
        return len(self.img_path_list)
