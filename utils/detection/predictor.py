# Author: Jintao Huang
# Time: 2020-6-6

import torch
from ..detection.utils import to


class Predictor:
    """$"""

    def __init__(self, model, device, labels=None):
        """

        :param model:
        :param device:
        :param labels: List
        """
        self.model = model.to(device)
        self.device = device
        self._pred_video_now = False
        self.labels = labels

    def pred(self, image):
        """

        :param image: Tensor[C, H, W]
        :return: target: str
        """
        self.model.eval()
        with torch.no_grad():
            x, _ = to([image], None, self.device)
            target = self.model(x)
        return self.labels[target]
