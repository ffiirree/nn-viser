import argparse
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from visor import *

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      type=str,   default='images/cat_dog.png')
    parser.add_argument('--label',      type=int,   default=243)
    parser.add_argument('--output-dir', type=str,   default='logs')
    args = parser.parse_args()

    print(args)

    model = torchvision.models.alexnet(pretrained=True).eval()
    hook = LayerBackwardHook(model, 0)

    image = Image.open(args.image)
    x = TF.normalize(TF.to_tensor(image), mean, std).unsqueeze(0).requires_grad_(True)

    model.zero_grad()
    output = model(x)

    one_hot_output = torch.zeros([1, 1000])
    one_hot_output[0][args.label] = 1

    output.backward(gradient=one_hot_output)

    gradients = hook.gradients.squeeze(0)

    image = TF.to_pil_image(torchvision.utils.make_grid(gradients, normalize=True))
    image.save(f'{args.output_dir}/grad_colorful.png')
    
    image = TF.to_pil_image(torchvision.utils.make_grid(torch.sum(torch.abs(gradients), dim=0),normalize=True))
    image.save(f'{args.output_dir}/grad_grayscale.png')

    image = TF.to_pil_image(torchvision.utils.make_grid(torch.sum(torch.abs(gradients * x.squeeze(0).detach()), dim=0),normalize=True))
    image.save(f'{args.output_dir}/grad_x_image.png')






