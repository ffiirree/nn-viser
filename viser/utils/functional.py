import torch
import torchvision.transforms.functional as TF

__all__ = ['normalize', 'denormalize', 'Denormalize']

def normalize(tensor):
    tensor = tensor.detach().clone()
    low, high = float(tensor.min()), float(tensor.max())
    tensor.sub_(low).div_(max(high - low, 1e-5))
    return tensor

def denormalize(tensor, mean, std, inplace=False, clamp=True):
        mean = [-m/s for m, s in zip(mean, std)]
        std = [1/s for s in std]

        tensor = TF.normalize(tensor, mean, std, inplace)
        # clamp to get rid of numerical errors
        return tensor if not clamp else torch.clamp(tensor, 0.0, 1.0)

class Denormalize(object):
    def __init__(self, mean, std, inplace=False, clamp=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.clamp = clamp

    def __call__(self, tensor):
        return denormalize(tensor, self.mean, self.std, self.inplace, self.clamp)
