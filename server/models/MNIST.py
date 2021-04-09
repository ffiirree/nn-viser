import torch
import torch.nn as nn

__all__ = ['MnistNet', 'ConvBlock', 'Generator']

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(0.05)
        )
    
    def forward(self, x):
        return self.conv(x)

# BN + Dropout: 99.66%
class MnistNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistNet, self).__init__()

        self.features = nn.Sequential(
            ConvBlock( 1, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 64, 3),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 64, 3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Generator(nn.Module):
    def __init__(self, num_classes, nz):
        super(Generator, self).__init__()

        self.embeds = nn.Embedding(num_classes, nz)

        ngf = 32

        self.net = nn.Sequential(
            # input : (batch_size, nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            # ConvBlock(ngf * 4, ngf * 4, padding=1),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 8, 4, 4)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # ConvBlock(ngf * 2, ngf * 2, padding=1),
            # nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 4, 8, 8)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # ConvBlock(ngf, ngf, padding=1),
            nn.ReLU(True),
            # state size: (batch_size, ngf * 2, 16, 16)
            # nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(opt.ngf),
            # nn.ReLU(True),
            # state size : (batch_size, ngf, 32, 32)
            nn.ConvTranspose2d(ngf, ngf, 4, 2, bias=False),
            ConvBlock(ngf, ngf),
            ConvBlock(ngf, ngf),
            nn.Conv2d(ngf, 1, kernel_size=3),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
            # state size : (batch_size, num_channels, 64, 64)
        )

    def forward(self, noise, labels):
        # x = torch.mul(self.embeds(labels), noise).reshape(noise.size()[0], noise.size()[1], 1, 1)
        # x = torch.add(self.embeds(labels), noise).reshape(noise.size()[0], noise.size()[1], 1, 1)
        x = self.embeds(labels).reshape(noise.size()[0], noise.size()[1], 1, 1)
        return self.net(x)



if __name__ == '__main__':
    x = torch.ones(64, 1, 28, 28)
    model = MnistNet(10)
    print(model(x).shape)