import torch
import argparse
import torchvision
import torch.nn as nn
import cvm
from cvm.utils import *
from viser.utils.utils import named_layers
from viser.hooks import LayersHook
from prettytable import PrettyTable
import timm
import torchvision.transforms.functional as TF
from PIL import Image
import sys
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--list-models', type=str, default=None)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--pth', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()

    if args.list_models:
        print(json.dumps(list_models(args.list_models), indent=4))
        sys.exit(0)
        
    model = create_model(args.model, pretrained=args.pretrained, cuda=False)
    model.eval()

    params_stat = PrettyTable()
    params_stat.field_names = ["name", "inp", "oup", "ks", 's',
                               "sum", "k-sum", "mean", "std"]
    params_stat.align = 'r'
    params_stat.float_format['min'] = ".3"
    params_stat.float_format['max'] = ".3"
    params_stat.float_format['k-sum'] = ".4"
    params_stat.float_format['sum'] = ".2"
    params_stat.float_format['mean'] = ".5"
    params_stat.float_format['std'] = ".4"

    for name, layer in named_layers(model):

        if isinstance(layer, (nn.Conv2d)):  # and layer.kernel_size[0] < 3:
            w = layer.weight
            params_stat.add_row([
                name.replace('trunk_output.block', ''),
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size[0],
                layer.stride[0],
                w.sum().item(),
                (w.sum() / w.shape[0]).item(),
                w.mean().item(),
                w.std().item()
            ])

        # elif isinstance(layer, nn.BatchNorm2d):
        #     w = layer.weight
        #     b = layer.bias
        #     params_stat.add_row([name.replace('trunk_output.block', ''), 'α', layer.num_features, '-', '-', w.sum().item(), '-', '-', w.mean().item(), '-', w.std().item()])
        #     params_stat.add_row([name.replace('trunk_output.block', ''), 'β', layer.num_features, '-', '-', b.sum().item(), '-', '-', b.mean().item(), '-', b.std().item()])

        # elif isinstance(layer, nn.Linear):
        #     params_stat.add_row([layer.in_features, layer.out_features, w.numel(), 1, w.min().item(), w.max().item(), w.sum().item(), (w.sum() / w.shape[0]).item(), 0, w.mean().item(), w.std().item()])
        #     # print(f'{layer.in_features:4d}-{layer.out_features:4d}]:',
        #     #       f'min={w.min():.3f}, max={w.max():.3f}, sum={w.sum():.2f}, mean={w.mean():7.5f}, std={w.std():.4f}')

    print(params_stat)

    # hook = LayersHook(model, types=(nn.Conv2d, nn.ReLU, nn.BatchNorm2d, nn.Linear))
    # features_stat = PrettyTable()
    # features_stat.field_names = ["name", "num", "min", "max", "f-sum", "mean", "std"]
    # features_stat.align = 'r'
    # features_stat.float_format['min'] = ".3"
    # features_stat.float_format['max'] = ".3"
    # features_stat.float_format['f-sum'] = ".3"
    # features_stat.float_format['sum'] = ".2"
    # features_stat.float_format['mean'] = ".4"
    # features_stat.float_format['std'] = ".4"

    # # input = torch.zeros(1, 3, 224, 224)

    # image = TF.to_tensor(Image.open("static/images/spider.png").convert('RGB'))
    # input = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)

    # with torch.inference_mode():
    #     model(input)

    # for i, act in enumerate(hook.activations):
    #     name = hook.layers[i].__class__.__name__
    #     if isinstance(hook.layers[i], nn.Conv2d):
    #         name = f'{hook.layers[i].__class__.__name__}_k{hook.layers[i].kernel_size[0]}_s{hook.layers[i].stride[0]}'
    #     features_stat.add_row([
    #         name,
    #         act.shape[1],
    #         act.min().item(),
    #         act.max().item(),
    #         (act.sum() / act.shape[1]).item(),
    #         act.mean().item(),
    #         act.std().item()
    #     ])

    # print(features_stat)
