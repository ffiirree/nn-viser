import argparse
import torch
import torchvision
import torchvision.transforms.functional as TF
from viser import *
from PIL import Image
import matplotlib.cm as cm
import numpy as np
from viser.attrs import GradCAM
from viser.utils import normalize

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

{ 56: 'images/snake.jpg', 243: 'images/cat_dog.png', 72: 'images/spider.png'}

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='alexnet')
    parser.add_argument('--image',      type=str,   default='images/cat_dog.png')
    parser.add_argument('--label',      type=int,   default=None)
    parser.add_argument('--output-dir', type=str,   default='logs')
    args = parser.parse_args()

    print(args)

    model = torchvision.models.alexnet(pretrained=True)
    
    image = Image.open(args.image).convert('RGB')
    x = TF.normalize(TF.to_tensor(image), mean, std).unsqueeze(0)
    
    grad_cam = GradCAM(model, 11)
    activations = grad_cam.attribute(x, 243).squeeze(0)

    cam = normalize(activations)

    grad_cam = TF.to_pil_image(cam).resize([224, 224], resample=Image.ANTIALIAS)
    grad_cam.save('xxxx.png')

    cmap = cm.get_cmap('hsv')
    heatmap = cmap(TF.to_tensor(grad_cam)[0].detach().numpy())
    Image.fromarray((heatmap * 255).astype(np.uint8)).save('heatmap.png')

    heatmap[:, :, 3] = 0.4

    heatmap_on_image = Image.new('RGBA', image.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, image.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, Image.fromarray((heatmap * 255).astype(np.uint8)))
    heatmap_on_image.save('heatmap_on_image.png')