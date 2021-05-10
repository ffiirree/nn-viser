import argparse
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from viser.utils import *
from viser.attrs import Saliency

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

{ 56: 'images/snake.jpg', 243: 'images/cat_dog.png', 72: 'images/spider.png'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='alexnet')
    parser.add_argument('--image',      type=str,   default='images/cat_dog.png')
    parser.add_argument('--label',      type=int,   default=243)
    parser.add_argument('--output-dir', type=str,   default='logs')
    args = parser.parse_args()

    print(args)

    if args.model == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    elif args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    else:
        raise ValueError('alexnet or vgg19')

    image = Image.open(args.image).convert('RGB')
    x = TF.normalize(TF.to_tensor(image), mean, std).unsqueeze(0)

    saliency = Saliency(model)
    attributions = saliency.attribute(x, args.label, abs=False).squeeze(0)

    # Vanilla gradients
    save_image(attributions, f'{args.output_dir}/grad_colorful_{args.label}.png')
    save_image(torch.sum(torch.abs(attributions), dim=0), f'{args.output_dir}/grad_grayscale_{args.label}.png')
    save_image(torch.sum(torch.abs(attributions * x.squeeze(0).detach()), dim=0), f'{args.output_dir}/grad_x_image_{args.label}.png')




