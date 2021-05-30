# Author: Jintao Huang
# Time: 2020-5-24

import pickle
import hashlib
import torch
import numpy as np
from torch.backends import cudnn
import os



def save_to_pickle(data, filepath):
    """$"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(filepath):
    """$"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def calculate_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            buffer = f.read(4096)
            if not buffer:
                break
            sha256.update(buffer)
    digest = sha256.hexdigest()
    return digest[:8]


def set_seed(seed=0):
    """网络重现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 取消cudnn加速时的省略精度产生的随机性
    cudnn.deterministic = True
    # cudnn.benchmark = True  # if benchmark == True, deterministic will be False


def save_params(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_params(model, filepath, prefix="", drop_layers=(), strict=True):
    """

    :param model: 变
    :param filepath: str
    :param prefix: 在pth的state_dict加上前缀.
    :param drop_layers: 对加完前缀后的pth进行剔除.
    :param strict: bool
    """

    load_state_dict = torch.load(filepath)
    # 1. 加前缀
    if prefix:
        for key in list(load_state_dict.keys()):
            load_state_dict[prefix + key] = load_state_dict.pop(key)
    # 2. drop
    for key in list(load_state_dict.keys()):
        for layer in drop_layers:
            if layer in key:
                load_state_dict.pop(key)
                break
    return model.load_state_dict(load_state_dict, strict)


def load_params_by_order(model, filepath, strict=True):
    """The parameter name of the pre-training model is different from the parameter name of the model"""
    load_state_dict = torch.load(filepath)
    # --------------------- 算法
    load_keys = list(load_state_dict.keys())
    model_keys = list(model.state_dict().keys())
    assert len(load_keys) == len(model_keys)
    # by order
    for load_key, model_key in zip(load_keys, model_keys):
        load_state_dict[model_key] = load_state_dict.pop(load_key)

    return model.load_state_dict(load_state_dict, strict)


# 选择执行设备
def select_device(device, batch_size=None):
    """copy from yolov5. https://github.com/ultralytics/yolov5"""
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'EfficientNet torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    print(s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def processing(x, target, transform=None):
    """
    :param x: numpy, shape[H, W, C]. RGB
    :return: Tensor, shape[C, H ,W]. RGB. 0-255
    """
    if transform is not None:
        x, target = transform(x, target)
    x = x[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to (C, H, W)
    x = np.ascontiguousarray(x)
    x = torch.from_numpy(x)
    return x, target
