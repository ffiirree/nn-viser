import torch
import torchvision.transforms.functional as F

def denormalize(tensor, mean, std, inplace=False, clamp=True):
        mean = [-m/s for m, s in zip(mean, std)]
        std = [1/s for s in std]

        tensor = F.normalize(tensor, mean, std, inplace)
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
