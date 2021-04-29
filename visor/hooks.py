import os
import json
import warnings
import torch
from torch import nn
import torchvision

__all__ = ['LayerForwardHook', 'LayerBackwardHook', 'ActivationsHook', 'FiltersHook']

class LayerForwardHook:
    def __init__(
        self,
        model: nn.Module,
        layer_index: int,
        types:tuple=(nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.AdaptiveAvgPool2d)
    ) -> None:
        self.model = model
        self.index = 0
        self.types = types
        self.layer = None
        self.filters = None

        for layer in self.model.modules():
            if isinstance(layer, self.types):
                if layer_index == self.index:
                    self.layer = layer
                    layer.register_forward_hook(self)
                    break

                self.index += 1

    def __call__(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        if isinstance(module, nn.Conv2d):
            self.filters = module.weight.detach()
        self.activations = output
        self.gradients = input

    def __str__(self) -> str:
        return { 'filters' : self.filters.shape, 'activations' : self.activations.shape }

class LayerBackwardHook:
    def __init__(
        self,
        model: nn.Module,
        layer_index: int,
        types:tuple=(nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.AdaptiveAvgPool2d)
    ) -> None:
        self.model = model
        self.index = 0
        self.types = types
        self.layer = None

        for layer in self.model.modules():
            if isinstance(layer, self.types):
                if layer_index == self.index:
                    self.layer = layer
                    layer.register_backward_hook(self)
                    break

                self.index += 1

    def __call__(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        self.gradients = grad_input[0]

    def __str__(self) -> str:
        return { 'gradients' : self.gradients.shape }

class ActivationsHook:
    def __init__(
        self,
        model:nn.Module,
        types:tuple=(nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.AdaptiveAvgPool2d),
        split_types:tuple=(nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d),
        stop_types:tuple=None,
    ) -> None:
        self.model = model
        self.activations = []
        self.index = 0
        self.types = types
        self.split_types = split_types
        self.stop_types = stop_types

        self.max = None
        self.min = None

        for layer in model.modules():
            if self.stop_types and isinstance(layer, self.stop_types):
                break

            if isinstance(layer, self.types):
                layer.register_forward_hook(self)

    def __call__(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if isinstance(module, self.split_types):
            self.activations.append({})
            self.index = len(self.activations) - 1

        self.max = max(self.max, output.max().item()) if self.max != None else output.max().item()
        self.min = min(self.min, output.min().item()) if self.min != None else output.min().item()

        if isinstance(module, nn.Conv2d):
            self.activations[self.index][f'conv2d_{self.index}'] = output.detach().clone()
        elif isinstance(module, nn.Linear):
            self.activations[self.index][f'fc_{self.index}'] = output.detach().clone()
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            self.activations[self.index][f'avg_{self.index}'] = output.detach().clone()
        elif isinstance(module, nn.ReLU):
            self.activations[self.index][f'relu_{self.index}'] = output.detach().clone()
        elif isinstance(module, nn.MaxPool2d):
            self.activations[self.index][f'max_pool2d_{self.index}'] = output.detach().clone()
        else:
            raise ValueError

    def clear(self):
        self.activations.clear()
        self.index = 0

    def __str__(self) -> str:
        return str([{ name : list(unit[name].shape) for name in unit} for unit in self.activations])

    def save(self, dir:str, split_channels:bool=False, normalization_scope:str='layer'):
        valid_scopes = {'channel', 'layer', 'unit', 'global'}

        if normalization_scope not in valid_scopes:
            raise ValueError(f"normalization_scope must be one of {valid_scopes}, but got normalization_scope='{normalization_scope}'")

        if not split_channels and normalization_scope == 'channel':
            normalization_scope = 'layer'
            warnings.warn("normalization_scope can't be 'channel' when split_channels is 'true'.", UserWarning)

        if not os.path.exists(dir):
            os.makedirs(dir)

        ret = {
            'scope': normalization_scope,
            'split': split_channels,
            'range': [self.min, self.max],
            'units': []
        }

        if not split_channels:
            for i, unit in enumerate(self.activations):
                ret['units'].append({'range': [], 'layers': {}})
                
                unit_low, unit_high = min([x.min().item() for x in unit.values()]), max([x.max().item() for x in unit.values()])
                ret['units'][i]['range'] = [unit_low, unit_high]

                for name in unit:
                    layer_low, layer_high = unit[name].min().item(), unit[name].max().item()

                    if normalization_scope == 'global':
                        low, high = self.min, self.max
                    elif normalization_scope == 'unit':
                        low, high = unit_low, unit_high
                    elif normalization_scope == 'layer':
                        low, high = layer_low, layer_high

                    filename = f'{dir}/activations_{name}.png'
                    torchvision.utils.save_image(unit[name].squeeze().unsqueeze(1), filename, normalize=True)
                    ret['units'][i]['layers'][name] = {'path': filename, 'range': [layer_low, layer_high]}
        else:
            for i, unit in enumerate(self.activations):
                ret['units'].append({'range': [], 'layers': {}})
                
                unit_low, unit_high = min([x.min().item() for x in unit.values()]), max([x.max().item() for x in unit.values()])
                ret['units'][i]['range'] = [unit_low, unit_high]

                for name in unit:
                    if unit[name].dim() < 4:
                        warnings.warn(f"can't save just a pixel as an image: {name}@{list(unit[name].squeeze().shape)}", UserWarning)
                    else:
                        ret['units'][i]['layers'][name] = { 'range':[], 'channels': [] }

                        layer_low, layer_high = unit[name].min().item(), unit[name].max().item()
                        ret['units'][i]['layers'][name]['range'] = [layer_low, layer_high]

                        for idx, activation in enumerate(unit[name].squeeze()):
                            if not os.path.exists(f'{dir}/activations_{name}'):
                                os.makedirs(f'{dir}/activations_{name}')

                            ch_low, ch_high = activation.min().item(), activation.max().item()

                            if normalization_scope == 'global':
                                low, high = self.min, self.max
                            elif normalization_scope == 'unit':
                                low, high = unit_low, unit_high
                            elif normalization_scope == 'layer':
                                low, high = layer_low, layer_high
                            elif normalization_scope == 'channel':
                                low, high = ch_low, ch_high

                            filename = f'{dir}/activations_{name}/{idx}.png'
                            torchvision.utils.save_image(activation.unsqueeze(0), filename, value_range=(low, high), normalize=True)

                            ret['units'][i]['layers'][name]['channels'].append({'path': filename, 'range': [ch_low, ch_high]})
        return ret #json.dumps(ret, indent=4, separators=(', ', ': '))

class FiltersHook:
    def __init__(self, model: nn.Module) -> None:
        self.filters = []
        self.index = 0

        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d)):
                layer.register_forward_hook(self)

    def __call__(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.index = len(self.filters)
        self.filters.append((f'conv2d_{self.index}.weight', module.weight.detach()))

    def __str__(self) -> str:
        return [(name, list(t.shape)) for name, t in self.filters].__str__()