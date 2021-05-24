import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import viser.utils
import os
import torchvision.models as models

__all__ = ['manual_seed', 'save_image', 'named_layers', 'torch_models', 'get_model']

def manual_seed(seed: int = 0):
    r"""
        https://pytorch.org/docs/stable/notes/randomness.html
        https://discuss.pytorch.org/t/random-seed-initialization/7854/20
    """
    # numpy
    np.random.seed(seed)
    # Python
    random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    # cuda
    torch.cuda.manual_seed(seed)
    # multi-gpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
def save_image(tensor: torch.Tensor, filename: str, normalize: bool = True) -> None:
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image = TF.to_pil_image(viser.utils.normalize(tensor) if normalize else tensor)
    image.save(filename)
    
    return filename

def named_layers(module, memo = None, prefix: str = ''):
    if memo is None:
        memo = set()
    if module not in memo:
        memo.add(module)
        if not module._modules.items():
            yield prefix, module
        for name, module in module._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in named_layers(module, memo, submodule_prefix):
                yield m

def torch_models():
    return [name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name])]

def get_model(name:str, pretrained:bool=True) -> nn.Module:
    return models.__dict__[name](pretrained=pretrained)