# Author: Jintao Huang
# Time: 2020-5-21

import numpy as np
import cv2 as cv
from PIL import Image


def imwrite(image, filename):
    """cv无法读取中文字符 (CV cannot read Chinese characters)"""
    retval, arr = cv.imencode('.' + filename.rsplit('.', 1)[1], image)  # retval: 是否保存成功
    if retval is True:
        arr.tofile(filename)
    return retval


def imread(filename):
    """cv无法读取中文字符 (CV cannot read Chinese characters)"""
    arr = np.fromfile(filename, dtype=np.uint8)
    return cv.imdecode(arr, -1)


def pil_to_cv(img):
    """转PIL.Image到cv (Turn PIL.Image to CV(BGR))

    :param img: PIL.Image. RGB, RGBA, L. const
    :return: ndarray. BGR, BGRA, L  (H, W, C{1, 3, 4})
    """
    mode = img.mode
    arr = np.asarray(img)
    if mode == "RGB":
        arr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)
    elif mode == "RGBA":
        arr = cv.cvtColor(arr, cv.COLOR_RGBA2BGRA)
    elif mode in ("L",):
        arr = arr
    else:
        raise ValueError("img.mode nonsupport")
    return arr


def cv_to_pil(arr):
    """转cv到PIL.Image (Turn CV(BGR) to PIL.Image)

    :param arr: ndarray. BGR, BGRA, L. const
    :return: PIL.Image. RGB, RGBA,L
    """

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        arr = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    else:  # 4
        arr = cv.cvtColor(arr, cv.COLOR_BGRA2RGBA)
    return Image.fromarray(arr)


def resize_max(image, max_height=None, max_width=None):
    """将图像resize成最大不超过max_height, max_width的图像. (双线性插值)

    :param image: PIL.Image / ndarray(H, W, C). BGR. const
    :param max_width: int
    :param max_height: int
    :return: shape(H, W, C). BGR"""

    # 1. 输入
    if isinstance(image, np.ndarray):
        height_ori, width_ori = image.shape[:2]
    elif isinstance(image, Image.Image):
        height_ori, width_ori = image.height, image.width
    else:
        raise ValueError("the type of image nonsupport. only support ndarray(H, W, 3) and PIL.Image")

    max_width = max_width or width_ori
    max_height = max_height or height_ori
    # 2. 算法
    width = max_width
    height = width / width_ori * height_ori
    if height > max_height:
        height = max_height
        width = height / height_ori * width_ori
    if isinstance(image, np.ndarray):
        image = cv.resize(image, (int(width), int(height)), interpolation=cv.INTER_LINEAR)
    elif isinstance(image, Image.Image):
        image = image.resize((int(width), int(height)), Image.BILINEAR)

    return image


def resize_equal(image, height=None, width=None, scale=None):
    """等比例缩放 (优先级: height > width > scale). (双线性插值)

    :param image: PIL.Image / ndarray(H, W, C) (BGR). const
    :param height: 高
    :param width: 宽
    :param scale: 比例
    """
    # 1. 输入
    if isinstance(image, np.ndarray):
        height_ori, width_ori = image.shape[:2]
    elif isinstance(image, Image.Image):
        height_ori, width_ori = image.height, image.width
    else:
        raise ValueError("the type of image nonsupport. only support ndarray(H, W, 3) and PIL.Image")
    # 2. 算法
    if height:
        width = height / height_ori * width_ori
    elif width:
        height = width / width_ori * height_ori
    elif scale:
        height, width = height_ori * scale, width_ori * scale
    else:
        ValueError("All of height, width, scale are `None`")

    if isinstance(image, np.ndarray):
        image = cv.resize(image, (int(width), int(height)), interpolation=cv.INTER_LINEAR)
    elif isinstance(image, Image.Image):
        image = image.resize((int(width), int(height)), Image.BILINEAR)
    return image
