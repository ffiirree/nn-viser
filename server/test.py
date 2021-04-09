import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

cmap=plt.get_cmap('RdBu')
print(cmap(23))


bwr = torch.zeros([256, 3], dtype=torch.uint8)

for index in range(bwr.shape[0]):
    bwr[index][2] = (32 / 3 * 5 + index) if index <= 160 else (256 - index) / 2 * 3
    bwr[index][1] = 2 * index   if index <= 127 else 511 - 2 * index
    bwr[index][0] = (100 + index / 3 * 2) if index <= 96 else (160 - (index - 96)) / 2 * 3

def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

def to_bwr(tensor):
    image = tensor.repeat(3, 1, 1)
    norm_ip(tensor, tensor.min(), tensor.max())
    tensor = tensor.mul(255).clamp_(0, 255).type(torch.uint8)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            idx = tensor[i][j]
            c = cmap(idx.item())
            image[0][i][j] = c[0]
            image[1][i][j] = c[1]
            image[2][i][j] = c[2]
    return image


x = torch.zeros([64, 64])
for i in range(64):
    for j in range(64):
        x[i][j] = (i + 1) / 64.0
# print(x.shape)     
# x = x.repeat(3, 1, 1)
# print(x.shape)
x = to_bwr(x)
torchvision.utils.save_image(x.detach(), 'a.png', normalize=True)
