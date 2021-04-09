from typing import Generator
from torch._C import device
import torch
import torchvision
from models import *
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator(10, 100)
model.load_state_dict(torch.load('logs/mnist_g_tiny.pth'))
model.to(device)

for i in range(10):
    g_labels = torch.tensor([i]).to(device)
    z = torch.mul(torch.randn(1, 100), 0.1).to(device)
    image = model(z, g_labels)
    print(image.shape)

    grid = torchvision.utils.make_grid(image.squeeze(0), normalize=True)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.convert('L').save(f"logs/g_{g_labels[0].item()}.png")

# torchvision.utils.save_image(image, "logs/g_0.png", normalize=True)