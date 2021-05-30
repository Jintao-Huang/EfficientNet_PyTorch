# Author: Jintao Huang
# Time: 2020-5-21

import numpy as np
import cv2 as cv
from PIL import Image
import random
import math


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

    :param image: ndarray[H, W, C]. BGR. const
    :param max_width: int
    :param max_height: int
    :return: ndarray[H, W, C]. BGR"""

    # 1. 输入
    height0, width0 = image.shape[:2]
    max_width = max_width or width0
    max_height = max_height or height0
    # 2. 算法
    ratio = min(max_height / height0, max_width / width0)
    new_shape = int(round(width0 * ratio)), int(round(height0 * ratio))
    image = cv.resize(image, new_shape, interpolation=cv.INTER_LINEAR)
    return image


def get_scale_pad(img_shape, new_shape, rect=True, stride=32, only_pad=False):
    """

    :param img_shape: Tuple[W, H]
    :param new_shape: Tuple[W, H]
    :param rect: True: 矩形, False: 正方形
    :param stride:
    :param only_pad:
    :return: ratio: float, new_unpad: Tuple[W, H], (pad_w, pad_h)
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = 1 if only_pad else min(new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
    new_unpad = int(round(img_shape[0] * ratio)), int(round(img_shape[1] * ratio))  # new image unpad shape

    # Compute padding
    pad_w, pad_h = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # square
    if rect:  # detect. rect
        pad_w, pad_h = pad_w % stride, pad_h % stride
    pad_w, pad_h = pad_w / 2, pad_h / 2  # divide padding into 2 sides
    return ratio, new_unpad, (pad_w, pad_h)


def resize_pad(img, new_shape=640, rect=True, stride=32, only_pad=False, fill_value=114):
    """copy from official yolov5 letterbox()

    :param img: ndarray[H, W, C]
    :param new_shape: Union[int, Tuple[W, H]]
    :param rect: bool. new_shape是否自动适应
    :param color: BRG
    :param stride: int
    :param only_pad: 不resize, 只pad
    :return: img: ndarray[H, W, C], ratio: float, pad: Tuple[W, H]
    """
    # Resize and pad image
    fill_value = (fill_value, fill_value, fill_value) if isinstance(fill_value, (int, float)) else fill_value
    shape = img.shape[1], img.shape[0]  # Tuple[W, H]
    new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
    ratio, new_unpad, (pad_w, pad_h) = get_scale_pad(shape, new_shape, rect, stride, only_pad)
    if ratio != 1:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))  # 防止0.5, 0.5
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=fill_value)  # add border(grey)
    return img, ratio, (pad_w, pad_h)  # 处理后的图片, 比例, padding的像素


def random_perspective(img, degrees=10, translate=.1, scale=.1, shear=10, perspective=0, fill_value=114):
    """torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    :param img: ndarray[H, W, C]. BGR
    :param degrees: 旋转
    :param translate: 平移
    :param scale: 缩放
    :param shear: 斜切
    :param perspective: 透视
    :return: ndarray[H, W, C]. BGR
    """
    #
    fill_value = (fill_value, fill_value, fill_value) if isinstance(fill_value, (int, float)) else fill_value
    height, width = img.shape[:2]
    # Center.
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective 透视
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale 旋转, 缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear 斜切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv.warpPerspective(img, M, dsize=(width, height), flags=cv.INTER_LINEAR,
                                     borderValue=fill_value)
        else:  # affine
            img = cv.warpAffine(img, M[:2], dsize=(width, height), flags=cv.INTER_LINEAR,
                                borderValue=fill_value)
    return img


def random_crop(image, scale_range, fill_value=114):
    """

    :param image: ndarray[H, W, C]. BGR
    :param scale_range: 裁剪范围. [2个值]. [hw_scale_min, hw_scale_max]
    :return: ndarray[H, W, C]. BGR
    """
    h0, w0 = image.shape[:2]
    h = int(random.uniform(scale_range[0], scale_range[1]) * h0)
    w = int(random.uniform(scale_range[0], scale_range[1]) * w0)
    left0, top0 = int(random.uniform(0, w0 - w)), int(random.uniform(0, h0 - h))
    left, top = (w0 - w) // 2, (h0 - h) // 2  # 在中心
    out = np.full_like(image, fill_value=fill_value)
    out[top:top + h, left: left + w] = image[top0:top0 + h, left0: left0 + w]
    return out


def augment_hsv(img, h=0.015, s=0.7, v=0.4):
    """

    :param img: ndarray[H, W, C]. BGR
    :param h: 色调
    :param s: 饱和度
    :param v: 明度
    :return:
    """
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1  # random gains
    hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
    img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)  # no return needed
    return img


def draw_box(image, box, color):
    """在给定图像上绘制一个方框 (Draws a box on a given image)

    :param image: shape(H, W, C) BGR. 变
    :param box: len(4), (ltrb)
    :param color: len(3). BGR
    """
    image = np.asarray(image, np.uint8)
    box = np.asarray(box, dtype=np.int)
    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2, cv.LINE_4)


def draw_text(image, box, text, rect_color):
    """在图像的方框上方绘制文字 (Draw text above the box of the image)

    :param image: shape(H, W, C) BGR. 变
    :param box: len(4), (ltrb)
    :param text: str
    :param rect_color: BGR
    """
    image = np.asarray(image, np.uint8)
    box = np.asarray(box, dtype=np.int)
    cv.rectangle(image, (box[0] - 1, box[1] - 16), (box[0] + len(text) * 9, box[1]), rect_color, -1, cv.LINE_4)
    cv.putText(image, text, (box[0], box[1] - 4), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_8)


def draw_target_in_image(image, boxes, labels, scores, color=(0, 252, 124)):
    """画框在image上 (draw boxes and text in image)

    :param image: ndarray[H, W, C]. BGR. not const
    :param boxes: ndarray[X, 4]. ltrb, 未归一化
    :param labels: List[str]. Len[X].
    :param scores: ndarray[X]. 从大到小排序
    :param color: List -> tuple(B, G, R)  # [0, 256).
    :return: None
    """
    boxes = np.round(boxes).astype(np.int32)

    # draw
    for box in boxes:
        draw_box(image, box, color=color)  # 画方框
    if labels is None:
        return
    if scores is None:
        scores = [None] * labels.shape[0]
    for box, label, score in reversed(list(zip(boxes, labels, scores))):  # 先画框再写字: 防止框把字盖住. 概率大盖住概率小
        text = "%s %.2f" % (label, score) if score else "%s" % label
        draw_text(image, box, text, color)  # 写字
