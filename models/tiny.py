import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MnistNetTiny']
class MnistNetTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d( 1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.avg = nn.AdaptiveAvgPool2d((1))

    def forward(self, x):
        return self.avg(self.features(x))