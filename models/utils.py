try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch
import torch.nn.functional as F
import math

def freeze_layers(model, layers):
    """冻结层"""
    for name, parameter in model.named_parameters():
        for layer in layers:
            if layer in name:  # 只要含有名字即可
                parameter.requires_grad_(False)
                break
        else:
            parameter.requires_grad_(True)


def model_info(model, img_size):
    img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
    num_params = sum(x.numel() for x in model.parameters())  # number parameters
    num_grads = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    try:  # FLOPS
        from thop import profile
        p = next(model.parameters())
        x = torch.rand((1, 3, 32, 32), dtype=p.dtype, device=p.device)
        macs, num_params = profile(model, inputs=(x,), verbose=False)
        flops = 2 * macs
        flops_str = ", %.1f GFLOPS" % (flops * img_size[0] * img_size[1] / 32 / 32 / 1e9)  # 640x640 GFLOPS
    except (ImportError, Exception):
        flops_str = ""

    print("Model Summary: %d layers, %d parameters, %d gradients%s" %
          (len(list(model.modules())), num_params, num_grads, flops_str))


def label_smoothing_cross_entropy(pred, target, smoothing: float = 0.1):
    """reference: https://github.com/seominseok0429/label-smoothing-visualization-pytorch

    :param pred: shape(N, In). 未过softmax
    :param target: shape(N,)
    :param smoothing: float
    :return: shape()
    """
    pred = F.log_softmax(pred, dim=-1)
    ce_loss = F.nll_loss(pred, target)
    smooth_loss = -torch.mean(pred)
    return (1 - smoothing) * ce_loss + smoothing * smooth_loss


def cosine_annealing_lr(epoch, T_max, min_lr, max_lr):
    return min_lr + (max_lr - min_lr) * (1 + math.cos(epoch / T_max * math.pi)) / 2