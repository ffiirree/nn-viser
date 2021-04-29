import argparse
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import AutoEncoder

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device = torch.device('cuda')

    model = AutoEncoder(28 * 28)
    if not args.train:
        model.load_state_dict(torch.load('autoencoder.pth'))
    model.to(device)

    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=(0.1307,), std=(0.3081,))
    ])


    if not args.train:
        test_data = torchvision.datasets.MNIST(
            train=False,
            download=True,
            root='./data',
            transform=transform
        )
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

        model.eval()
        x = [[] for i in range(10)]
        y = [[] for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            pre = model.encoder(images.flatten(2)).squeeze().cpu().detach().numpy()
            x[labels.item()].append(pre[0])
            y[labels.item()].append(pre[1])

        
        for i in range(10):
            plt.scatter(x[i], y[i], s=2)
        plt.show()

    else:
        train_data = torchvision.datasets.MNIST(
            train=True,
            download=True,
            root='./data',
            transform=transform
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(25):
            model.train()

            for i, (images, _) in enumerate(train_loader):
                images = images.to(device)

                optimizer.zero_grad()
                pre = model(images)
                loss = mse_loss(pre, images)
                print(f'epoch: #{epoch:>2}] loss = {loss.item()}')
                loss.backward()
                optimizer.step()

                if(i != 0 and i % 100 == 0):
                    torchvision.utils.save_image(pre, 'logs/pre_mnist.png')

        
        torch.save(model.state_dict(), 'autoencoder.pth')