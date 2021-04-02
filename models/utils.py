try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def freeze_layers(model, layers):
    """冻结层"""
    for name, parameter in model.named_parameters():
        for layer in layers:
            if layer in name:  # 只要含有名字即可
                parameter.requires_grad_(False)
                break
        else:
            parameter.requires_grad_(True)
