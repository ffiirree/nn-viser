import argparse
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from viser.hooks import *

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
def backward_op(module, grad_input, grad_output, activations):
    activation = activations[-1]
    activation[activation > 0] = 1
    modified_grad_output = activation * torch.clamp(grad_input[0], min=0.0)
    del activations[-1]
    return (modified_grad_output, )

{ 56: 'images/snake.jpg', 243: 'images/cat_dog.png', 72: 'images/spider.png'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='alexnet')
    parser.add_argument('--image',      type=str,   default='images/spider.png')
    parser.add_argument('--label',      type=int,   default=None)
    parser.add_argument('--output-dir', type=str,   default='logs')
    args = parser.parse_args()

    print(args)

    if args.model == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    elif args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    else:
        raise ValueError('alexnet or vgg19')
    
    model.eval()

    hook = LayerHook(model, backward_op=backward_op)
    bp_hook = LayerBackwardHook(model, 0)

    image = Image.open(args.image).convert('RGB')
    x = TF.normalize(TF.to_tensor(image), mean, std).unsqueeze(0).requires_grad_(True)

    model.zero_grad()
    output = torch.softmax(model(x), dim=1)

    if args.label:
        output[0, args.label].backward()
    else:
        output.max().backward()

    gradients = bp_hook.gradients.squeeze(0)

    # Guided gradients
    image = TF.to_pil_image(torchvision.utils.make_grid(gradients, normalize=True))
    image.save(f'{args.output_dir}/guided_grad_colorful_{args.label}.png')
    
    # Guided saliecncy Maps
    image = TF.to_pil_image(torchvision.utils.make_grid(torch.sum(torch.abs(gradients), dim=0), normalize=True))
    image.save(f'{args.output_dir}/guided_grad_grayscale_{args.label}.png')

    # Guided Gradients x Input
    image = TF.to_pil_image(torchvision.utils.make_grid(torch.abs(gradients * x.squeeze(0).detach()), normalize=True))
    image.save(f'{args.output_dir}/guided_grad_x_image_colorful_{args.label}.png')

    image = TF.to_pil_image(torchvision.utils.make_grid(torch.sum(torch.abs(gradients * x.squeeze(0).detach()), dim=0), normalize=True))
    image.save(f'{args.output_dir}/guided_grad_x_image_{args.label}.png')






