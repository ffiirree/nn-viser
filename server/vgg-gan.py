import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from MNIST import *

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of the generator filters')

opt = parser.parse_args()

dataset = torchvision.datasets.MNIST(
    train=True,
    download=True,
    root=os.path.expanduser('~/data/datasets/'),
    transform=transforms.Compose([
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g = Generator(10, 32)
    d = MnistNet(num_classes=11)
    d.load_state_dict(torch.load('logs/mnist_11.pth'))

    g.to(device=device)
    d.to(device=device)

    aux_loss = nn.CrossEntropyLoss()
    optimizer_g = optim.SGD(g.parameters(), lr=0.00001, momentum=0.9)

    batch_size = 64

    for i in range(100000):
        labels = torch.tensor([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 2, 3]).to(device)
        z = torch.mul(torch.randn(batch_size, opt.nz), 0.1).to(device)

        g.zero_grad()
        x = g(z, labels)
        # x = (x > 0.5).float()
        # print(x.shape)
        pre_labels = d(x)
        maxs = torch.max(pre_labels, 1)
        # print(maxs.values[0:10], maxs.indices[0:10])
        loss = aux_loss(pre_labels, labels)
        loss.backward()
        print(f'loss.item()\t{maxs.indices[0:10]}')
        optimizer_g.step()
        
        torchvision.utils.save_image(x.detach(), 'logs/fake.png', normalize=True)
        time.sleep(0.1)

    # print(z)
    # print(torch.mul(g.embeds(labels), z))

    # print(labels)
