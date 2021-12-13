import numpy as np
import random
import torch
import torchvision
import torchvision.transforms.functional as TF
import viser.utils
import os
from PIL import Image
import matplotlib.cm as cm
import timm

__all__ = ['manual_seed', 'save_image', 'named_layers',
           'save_RdBu_image']


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


def read_image(filename: str, mode: str = 'RGB') -> torch.Tensor:
    return TF.to_tensor(Image.open(filename).convert(mode)).mul_(255)


def save_image(tensor: torch.Tensor, filename: str, normalize: bool = True) -> None:
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image = TF.to_pil_image(viser.utils.normalize(
        tensor) if normalize else tensor)
    image.save(filename)

    return filename


def save_RdBu_image(filename: str, image: torch.Tensor, range: torch.Tensor = None):

    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # -abs_max, abs_max
    abs_max = range if range else max(abs(image.min()), abs(image.max()))
    image = viser.utils.normalize(image, -abs_max, abs_max)

    # RdBu
    image = torch.from_numpy(cm.get_cmap('RdBu')(image.detach().cpu().numpy()))
    image = image.permute((0, 3, 1, 2))

    torchvision.utils.save_image(
        image, filename, nrow=1, padding=0, normalize=False)
    return filename


def named_layers(module, memo=None, prefix: str = ''):
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
