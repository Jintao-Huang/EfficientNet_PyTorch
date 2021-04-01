# Author: Jintao Huang
# Time: 2020-6-6

import torch.utils.data as tud
import torch
from PIL import Image
import torchvision.transforms as trans
from ..utils import load_from_pickle


def get_dataset_from_pickle(pkl_path, transforms=None):
    image_path_list, target_list = load_from_pickle(pkl_path)
    return MyDataset(image_path_list, target_list, transforms)


class MyDataset(tud.Dataset):
    def __init__(self, image_path_list, target_list, transform=None):
        """

        :param image_path_list: List[str]
        :param target_list: List[int]
        :param transform: func(image: PIL.Image, target: ) -> image: Tensor[C, H, W] RGB, targets
            默认(self._default_trans_func)
        """
        assert len(image_path_list) == len(target_list)
        self.image_path_list = image_path_list
        self.target_list = target_list
        self.transform = transform or self._default_transform

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        target = self.target_list[idx]

        if isinstance(idx, slice):
            return self.__class__(image_path, target, self.transform)
        else:
            with Image.open(image_path) as image:
                image = self.transform(image)
            return image, target

    def __len__(self):
        return len(self.image_path_list)

    @staticmethod
    def _default_transform(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = trans.ToTensor()(image)
        return image
