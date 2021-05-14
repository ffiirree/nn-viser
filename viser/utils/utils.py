import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import viser.utils
import os

__all__ = ['manual_seed', 'save_image']

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