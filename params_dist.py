import torch
import argparse
import torch.nn as nn
import cvm
from cvm.models.core import blocks
from viser.utils.utils import named_layers
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--model', '-m', type=str)

    args = parser.parse_args()
    print(args)
    
    if args.list_models:
        print(json.dumps(cvm.utils.list_models(args.list_models), indent=4))
        sys.exit(0)

    model = cvm.utils.create_model(args.model, pretrained=True, cuda=False)

    model.eval()
    
    
    plt.ion()
    fig = plt.figure()
    # plt.xlim((-1, 1))
    ax = plt.subplot()
    # ax.axis([-1, 1, -1, 1])
    # ax.axis('equal')
    for name, layer in named_layers(model):
        if isinstance(layer, (nn.Conv2d, blocks.GaussianFilter, blocks.PointwiseBlock)):# and layer.kernel_size[0] > 1:
            w = layer.weight.flatten(0).detach().numpy()
            y = torch.normal(w.mean(), w.std(), w.shape).flatten(0).detach().numpy()
            plt.title(f'{name}_k{layer.kernel_size[0]}_s{layer.stride[0]}')
            
            
            # ax.scatter(w, y, s=1)
            ax.hist(w, bins=100, density=True)
            plt.pause(2)
            # break
    plt.pause(20)
            