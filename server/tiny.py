import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MnistNetTiny']

class MnistNetTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d( 1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv5 = nn.Conv2d(16, num_classes, kernel_size=3)

        self.avg = nn.AdaptiveAvgPool2d((1))

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1, inplace=True)

        conv2 = self.conv2(relu1)
        relu2 = F.relu(conv2, inplace=True)

        conv3 = self.conv3(relu2)
        relu3 = F.relu(conv3, inplace=True)

        conv4 = self.conv4(relu3)
        relu4 = F.relu(conv4, inplace=True)

        conv5 = self.conv5(relu4)
        relu5 = F.relu(conv5, inplace=True)

        avg = self.avg(relu5)

        return avg.squeeze()

if __name__ == '__main__':
    x = torch.ones(64, 1, 28, 28)
    model = MnistNetTiny(10)
    print(model(x).shape)