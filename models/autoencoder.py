import torch
import torch.nn as nn

__all__ = ['AutoEncoder']

class AutoEncoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.n_channels = n_channels

        self.encoder = nn.Sequential(
            nn.Linear(self.n_channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x.flatten(2))).reshape(x.shape)