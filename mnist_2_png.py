import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

if __name__ == '__main__':
    test_data = torchvision.datasets.MNIST(
        train=False,
        download=True,
        root='./data',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    for index, (image, mask) in enumerate(test_loader):
        grid = torchvision.utils.make_grid(image, normalize=True)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.convert('L').save(f'static/images/mnist/m_{index}.png')