# Author: Jintao Huang
# Time: 2020-5-18


def to(images, targets, device):
    """

    :param images: List[Tensor[C, H, W]]
    :param targets: List[int] / None
    :param device: str / device
    :return: images: List[Tensor[C, H, W]], targets: List[Tensor_int] / None
    """
    images = images.to(device)
    if targets is not None:
        targets = targets.to(device)
    return images, targets
