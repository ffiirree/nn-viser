import os
import sys
import warnings

from viser.utils import *
import torch
from torch import nn
import torchvision
import matplotlib.cm as cm

from cvm.models.core import blocks
from typing import List

__all__ = ['LayerHook', 'LayersHook',
           'ActivationsHook', 'GradientsHook', 'FiltersHook', 'StandardKernel']


logger = make_logger()


class LayerHook:
    def __init__(
        self,
        model: nn.Module,
        index: int = 0
    ):
        self.model = model
        self.index = index
        self.layer = None
        self.layer_name = None
        self.filters = None
        self.activations = None
        self.gradients = None

        self.layer_name, self.layer = list(
            named_layers(self.model))[self.index]
        self.layer.register_forward_hook(self.forward_hook)
        self.layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if isinstance(module, nn.Conv2d):
            self.filters = module.weight
        self.activations = output

    def backward_hook(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        self.gradients = grad_input[0]

    def __str__(self) -> str:
        return f'LayerHook(name: {self.layer_name}, index: {self.index}, layer: {self.layer})'


class LayersHook:
    def __init__(
        self,
        model: nn.Module,
        types: tuple = (nn.ReLU),
        forward_op=None,
        backward_op=None
    ) -> None:
        self.model = model
        self.types = types
        self.layers = []
        self.activations = []
        self.gradients = []
        self.forward_op = forward_op
        self.backward_op = backward_op

        for name, layer in named_layers(model):
            if isinstance(layer, self.types):
                layer.register_forward_hook(self.forward_hook)
                layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.activations.append(output.detach().clone())
        self.layers.append(module)
        if callable(self.forward_op):
            return self.forward_op(module, input, output)

    def backward_hook(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        self.gradients.append(grad_input[0])
        if callable(self.backward_op):
            return self.backward_op(module, grad_input, grad_output, self.activations)


class ActivationsHook:
    def __init__(
        self,
        model: nn.Module,
        split_types: tuple = (nn.Conv2d, nn.Linear),
        stop_types: tuple = None,
        hook_layers: int = 0
    ):
        self.model = model
        self.activations = []
        self.index = 0
        self.split_types = split_types
        self.stop_types = stop_types

        self.max = -sys.float_info.max
        self.min = sys.float_info.max

        for i, (_, layer) in enumerate(named_layers(model)):
            if hook_layers != 0 and i >= hook_layers:
                break

            if self.stop_types and isinstance(layer, self.stop_types):
                break

            layer.register_forward_hook(self)

    def __call__(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if isinstance(module, self.split_types):
            self.activations.append({})
            self.index = len(self.activations) - 1

        if isinstance(module, blocks.DepthwiseConv2d):
            self.activations[self.index][f'dwconv2d_{self.index}'] = output.detach(
            ).clone()
            self.max = max(self.max, output.max().item())
            self.min = min(self.min, output.min().item())
        elif isinstance(module, blocks.PointwiseConv2d):
            self.activations[self.index][f'pwconv2d_{self.index}'] = output.detach(
            ).clone()
            self.max = max(self.max, output.max().item())
            self.min = min(self.min, output.min().item())
        elif isinstance(module, nn.Conv2d):
            self.activations[self.index][f'conv2d_{self.index}'] = output.detach(
            ).clone()
            self.max = max(self.max, output.max().item())
            self.min = min(self.min, output.min().item())
        elif isinstance(module, nn.Linear):
            self.activations[self.index][f'fc_{self.index}'] = output.detach(
            ).clone()
            self.max = max(self.max, output.max().item())
            self.min = min(self.min, output.min().item())
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
            self.activations[self.index][f'relu_{self.index}'] = output.detach(
            ).clone()
            self.max = max(self.max, output.max().item())
            self.min = min(self.min, output.min().item())

        elif isinstance(module, nn.AdaptiveAvgPool2d):
            self.activations[self.index][f'avg_{self.index}'] = output.detach(
            ).clone()
            self.max = max(self.max, output.max().item())
            self.min = min(self.min, output.min().item())
        else:
            warnings.warn(f'{module.__class__.__name__}')

    def clear(self):
        self.activations.clear()
        self.index = 0

    def __isub__(self, other):
        for i, unit in enumerate(self.activations):
            for name in unit:
                self.activations[i][name] -= other.activations[i][name]
        return self

    def __str__(self) -> str:
        return str([{name: list(unit[name].shape) for name in unit} for unit in self.activations])

    def save(self, dir: str, split_channels: bool = False, normalization_scope: str = 'layer'):
        valid_scopes = {'channel', 'layer', 'unit', 'global'}

        if normalization_scope not in valid_scopes:
            raise ValueError(
                f"normalization_scope must be one of {valid_scopes}, but got normalization_scope='{normalization_scope}'")

        if not split_channels and normalization_scope == 'channel':
            normalization_scope = 'layer'
            warnings.warn(
                "normalization_scope can't be 'channel' when split_channels is 'true'.", UserWarning)

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

                unit_low, unit_high = min([x.min().item() for x in unit.values()]), max(
                    [x.max().item() for x in unit.values()])
                ret['units'][i]['range'] = [unit_low, unit_high]

                for name in unit:
                    layer_low, layer_high = unit[name].min(
                    ).item(), unit[name].max().item()

                    if normalization_scope == 'global':
                        low, high = self.min, self.max
                    elif normalization_scope == 'unit':
                        low, high = unit_low, unit_high
                    elif normalization_scope == 'layer':
                        low, high = layer_low, layer_high

                    filename = f'{dir}/activations_{name}.png'
                    image = unit[name].squeeze().unsqueeze(1)

                    cmap = cm.get_cmap('bwr')
                    heatmap = cmap(image.detach().numpy())
                    image = torch.from_numpy(heatmap)
                    # Image.fromarray((heatmap * 255).astype(np.uint8)).save(colorful_filename)

                    torchvision.utils.save_image(
                        image, filename, normalize=True)
                    ret['units'][i]['layers'][name] = {
                        'path': filename, 'range': [layer_low, layer_high]}
        else:
            for i, unit in enumerate(self.activations):
                ret['units'].append({'range': [], 'layers': {}})

                unit_low, unit_high = min([x.min().item() for x in unit.values()]), max(
                    [x.max().item() for x in unit.values()])
                ret['units'][i]['range'] = [unit_low, unit_high]

                for name in unit:
                    if unit[name].dim() < 4:
                        warnings.warn(
                            f"can't save just a pixel as an image: {name}@{list(unit[name].squeeze().shape)}", UserWarning)
                    else:
                        ret['units'][i]['layers'][name] = {
                            'range': [], 'channels': []}

                        layer_low, layer_high = unit[name].min(
                        ).item(), unit[name].max().item()
                        ret['units'][i]['layers'][name]['range'] = [
                            layer_low, layer_high]

                        for idx, activation in enumerate(unit[name].squeeze(0)):
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
                            _max = max(abs(low), abs(high))
                            image = normalize(activation, -_max, _max)
                            cmap = cm.get_cmap('RdBu')
                            heatmap = cmap(image.detach().numpy())
                            image = torch.from_numpy(
                                heatmap).permute((2, 0, 1))
                            # Image.fromarray((heatmap * 255).astype(np.uint8)).save(colorful_filename)
                            torchvision.utils.save_image(
                                image, filename, normalize=False)

                            ret['units'][i]['layers'][name]['channels'].append(
                                {'path': filename, 'range': [ch_low, ch_high]})
        return ret


class GradientsHook:
    def __init__(
        self,
        model: nn.Module,
        split_types: tuple = (nn.ReLU, nn.AdaptiveAvgPool2d, nn.Linear),
        stop_types: tuple = None,
    ) -> None:
        self.model = model
        self.gradients = []
        self.index = 0
        self.split_types = split_types
        self.stop_types = stop_types

        self.max = -sys.float_info.max
        self.min = sys.float_info.max

        for _, layer in named_layers(model):
            if self.stop_types and isinstance(layer, self.stop_types):
                break

            layer.register_backward_hook(self)

    def __call__(self, module: nn.Module, grad_wrt_input: torch.Tensor, grad_wrt_output: torch.Tensor):

        if isinstance(module, self.split_types):
            self.gradients.append({})
            self.index = len(self.gradients) - 1

        grad = grad_wrt_input[0]

        self.max = max(self.max, grad.max().item())
        self.min = min(self.min, grad.min().item())

        if isinstance(module, nn.Conv2d):
            self.gradients[self.index][f'conv2d_{self.index}'] = grad.detach(
            ).clone()
        elif isinstance(module, nn.Linear):
            self.gradients[self.index][f'fc_{self.index}'] = grad.detach(
            ).clone()
        elif isinstance(module, nn.ReLU):
            self.gradients[self.index][f'relu_{self.index}'] = grad.detach(
            ).clone()
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            self.gradients[self.index][f'avg_{self.index}'] = grad.detach(
            ).clone()
        else:
            warnings.warn(f'{module}')

    def clear(self):
        self.gradients.clear()
        self.index = 0

    def __isub__(self, other):
        for i, unit in enumerate(self.gradients):
            for name in unit:
                self.gradients[i][name].sub_(other.gradients[i][name])

        return self

    def __str__(self) -> str:
        return str([{name: list(unit[name].shape) for name in unit} for unit in self.gradients])

    def save(self, dir: str, split_channels: bool = False, normalization_scope: str = 'layer'):
        valid_scopes = {'channel', 'layer', 'unit', 'global'}

        if normalization_scope not in valid_scopes:
            raise ValueError(
                f"normalization_scope must be one of {valid_scopes}, but got normalization_scope='{normalization_scope}'")

        if not split_channels and normalization_scope == 'channel':
            normalization_scope = 'layer'
            warnings.warn(
                "normalization_scope can't be 'channel' when split_channels is 'true'.", UserWarning)

        if not os.path.exists(dir):
            os.makedirs(dir)

        ret = {
            'scope': normalization_scope,
            'split': split_channels,
            'range': [self.min, self.max],
            'units': []
        }

        if not split_channels:
            for i, unit in enumerate(self.gradients):
                ret['units'].append({'range': [], 'layers': {}})

                unit_low, unit_high = min([x.min().item() for x in unit.values()]), max(
                    [x.max().item() for x in unit.values()])
                ret['units'][i]['range'] = [unit_low, unit_high]

                for name in unit:
                    layer_low, layer_high = unit[name].min(
                    ).item(), unit[name].max().item()

                    if normalization_scope == 'global':
                        low, high = self.min, self.max
                    elif normalization_scope == 'unit':
                        low, high = unit_low, unit_high
                    elif normalization_scope == 'layer':
                        low, high = layer_low, layer_high

                    filename = f'{dir}/gradients_{name}.png'
                    image = unit[name].squeeze().unsqueeze(1)

                    cmap = cm.get_cmap('bwr')
                    heatmap = cmap(image.detach().numpy())
                    image = torch.from_numpy(heatmap)
                    # Image.fromarray((heatmap * 255).astype(np.uint8)).save(colorful_filename)

                    torchvision.utils.save_image(
                        image, filename, normalize=True)
                    ret['units'][i]['layers'][name] = {
                        'path': filename, 'range': [layer_low, layer_high]}
        else:
            for i, unit in enumerate(self.gradients):
                ret['units'].append({'range': [], 'layers': {}})

                unit_low, unit_high = min([x.min().item() for x in unit.values()]), max(
                    [x.max().item() for x in unit.values()])
                ret['units'][i]['range'] = [unit_low, unit_high]

                for name in unit:
                    if unit[name].dim() < 4:
                        warnings.warn(
                            f"can't save just a pixel as an image: {name}@{list(unit[name].squeeze().shape)}", UserWarning)
                    else:
                        ret['units'][i]['layers'][name] = {
                            'range': [], 'channels': []}

                        layer_low, layer_high = unit[name].min(
                        ).item(), unit[name].max().item()
                        ret['units'][i]['layers'][name]['range'] = [
                            layer_low, layer_high]

                        for idx, activation in enumerate(unit[name].squeeze(0)):
                            if not os.path.exists(f'{dir}/gradients_{name}'):
                                os.makedirs(f'{dir}/gradients_{name}')

                            ch_low, ch_high = activation.min().item(), activation.max().item()

                            if normalization_scope == 'global':
                                low, high = self.min, self.max
                            elif normalization_scope == 'unit':
                                low, high = unit_low, unit_high
                            elif normalization_scope == 'layer':
                                low, high = layer_low, layer_high
                            elif normalization_scope == 'channel':
                                low, high = ch_low, ch_high

                            filename = f'{dir}/gradients_{name}/{idx}.png'
                            _max = max(abs(low), abs(high))
                            image = normalize(activation, -_max, _max)
                            cmap = cm.get_cmap('RdBu')
                            heatmap = cmap(image.detach().numpy())
                            image = torch.from_numpy(
                                heatmap).permute((2, 0, 1))
                            # Image.fromarray((heatmap * 255).astype(np.uint8)).save(colorful_filename)
                            torchvision.utils.save_image(
                                image, filename, normalize=False)

                            ret['units'][i]['layers'][name]['channels'].append(
                                {'path': filename, 'range': [ch_low, ch_high]})
        return ret  # json.dumps(ret, indent=4, separators=(', ', ': '))


class FiltersHook:
    def __init__(self, model: nn.Module, stride: bool = 1, size: int = 0) -> None:
        self.filters = []
        self.index = 1

        for _, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, blocks.GaussianFilter, blocks.FixedConv2d)):
                if (layer.stride[0] == stride or stride == 0) and (layer.kernel_size[0] == size or size == 0):
                    self.filters.append(
                        (f'{self.index:02d}S{layer.stride[0]}K{layer.out_channels:03d}', layer.weight.detach()))
                self.index += 1

    @staticmethod
    def _save_image(name: str, filter: torch.Tensor, range: torch.Tensor = None):

        if filter.shape[-1] != 1:
            for kernel in filter:
                if kernel.flatten(0)[kernel.numel() // 2] < 0:
                    kernel.mul_(-1)

        filename = f"{name}.png"
        save_RdBu_image(filename, filter, range=range)
        return filename

    def save(self, dir: str):
        if not os.path.exists(dir):
            os.makedirs(dir)

        res = []
        for i, (name, filters) in enumerate(self.filters):
            logger.info(f'saving layer[{i:2d}/{len(self.filters)}]: {name}')
            res.append({'name': name, "filters": []})
            range = None# max(abs(filters.min()), abs(filters.max()))
            for idx, filter in enumerate(filters):
                res[i]['filters'].append(
                    self._save_image(
                        f'{dir}/{name}/{idx}',
                        filter,
                        range
                    )
                )
                res[i]['size'] = filter.shape[-1]
        logger.info('saved')
        return res

    def __str__(self) -> str:
        return [(name, list(t.shape)) for name, t in self.filters].__str__()


class StandardKernel:
    def __init__(self):
        self._kernels = []

        identity = torch.tensor(
            [[[
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ]]], dtype=torch.float32
        )

        sharpness = torch.tensor(
            [[[
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]], [[
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]], [[
                [-1, 0, -1],
                [0, 5, 0],
                [-1, 0, -1]
            ]]], dtype=torch.float32
        )

        edge = torch.tensor(
            [[[
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ]], [[
                [-1, -2, -3],
                [0, 0, 0],
                [1, 2, 3]
            ]], [[
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]], [[
                [-1, -3, -2],
                [0, 0, 0],
                [1, 3, 2]
            ]], [[
                [-1, 0, 0],
                [0, 2, 0],
                [0, 0, -1]
            ]], [[
                [0, -1, 0],
                [0,  2, 0],
                [0, -1, 0]
            ]], [[
                [0, 0, 0],
                [-1,  -1, 2],
                [0, 0, 0]
            ]], [[
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]]], dtype=torch.float32
        )

        embossing = torch.tensor(
            [[[
                [-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1]
            ]], [[
                [0, -1, 0],
                [-1, 0, 1],
                [0, 1, 0]
            ]], [[
                [2, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]]], dtype=torch.float32
        )

        box = torch.tensor(
            [[[
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]
            ]], [[
                [.0, 1/5, .0],
                [1/5, 1/5, 1/5],
                [.0, 1/5, .0]
            ]]], dtype=torch.float32
        )

        guassian = torch.tensor(
            [[[
                [1.0000, 1.5117, 1.0000],
                [1.5117, 2.2852, 1.5117],
                [1.0000, 1.5117, 1.0000]
            ]], [[
                [1.0000, 2.1842, 1.0000],
                [2.1842, 4.7707, 2.1842],
                [1.0000, 2.1842, 1.0000]
            ]], [[
                [1.0000, 2.7743, 1.0000],
                [2.7743, 7.6969, 2.7743],
                [1.0000, 2.7743, 1.0000]
            ]], [[
                [1.0000,  4.0104,  1.0000],
                [4.0104, 16.0832,  4.0104],
                [1.0000,  4.0104,  1.0000]
            ]]], dtype=torch.float32
        )

        guassian4 = torch.tensor(
            [[[
                [1.0000, 1.2840, 1.2840, 1.0000],
                [1.2840, 1.6487, 1.6487, 1.2840],
                [1.2840, 1.6487, 1.6487, 1.2840],
                [1.0000, 1.2840, 1.2840, 1.0000]
            ]], [[
                [1.0000, 1.5596, 1.5596, 1.0000],
                [1.5596, 2.4324, 2.4324, 1.5596],
                [1.5596, 2.4324, 2.4324, 1.5596],
                [1.0000, 1.5596, 1.5596, 1.0000]
            ]], [[
                [1.0000, 2.7183, 2.7183, 1.0000],
                [2.7183, 7.3891, 7.3891, 2.7183],
                [2.7183, 7.3891, 7.3891, 2.7183],
                [1.0000, 2.7183, 2.7183, 1.0000]
            ]], [[
                [1.0000,  5.9167,  5.9167,  1.0000],
                [5.9167, 35.0073, 35.0073,  5.9167],
                [5.9167, 35.0073, 35.0073,  5.9167],
                [1.0000,  5.9167,  5.9167,  1.0000]
            ]]]
        )

        guassian5 = torch.tensor(
            [[[
                [1.0000, 1.4550, 1.6487, 1.4550, 1.0000],
                [1.4550, 2.1170, 2.3989, 2.1170, 1.4550],
                [1.6487, 2.3989, 2.7183, 2.3989, 1.6487],
                [1.4550, 2.1170, 2.3989, 2.1170, 1.4550],
                [1.0000, 1.4550, 1.6487, 1.4550, 1.0000]
            ]], [[
                [1.0000, 1.9477, 2.4324, 1.9477, 1.0000],
                [1.9477, 3.7937, 4.7377, 3.7937, 1.9477],
                [2.4324, 4.7377, 5.9167, 4.7377, 2.4324],
                [1.9477, 3.7937, 4.7377, 3.7937, 1.9477],
                [1.0000, 1.9477, 2.4324, 1.9477, 1.0000]
            ]], [[
                [1.0000,  3.4545,  5.2221,  3.4545,  1.0000],
                [3.4545, 11.9334, 18.0395, 11.9334,  3.4545],
                [5.2221, 18.0395, 27.2699, 18.0395,  5.2221],
                [3.4545, 11.9334, 18.0395, 11.9334,  3.4545],
                [1.0000,  3.4545,  5.2221,  3.4545,  1.0000]
            ]], [[
                [1.0000,   6.3716,  11.8122,   6.3716,   1.0000],
                [6.3716,  40.5974,  75.2629,  40.5974,   6.3716],
                [11.8122,  75.2629, 139.5289,  75.2629,  11.8122],
                [6.3716,  40.5974,  75.2629,  40.5974,   6.3716],
                [1.0000,   6.3716,  11.8122,   6.3716,   1.0000]]
            ]], dtype=torch.float32
        )

        motion = torch.tensor(
            [[[
                [1/3, 0, 0],
                [0, 1/3, 0],
                [0, 0, 1/3]
            ]]], dtype=torch.float32
        )

        self._kernels = torch.cat(
            [sharpness, edge, embossing, box, motion], dim=0)

    def kernels(self):
        return self._kernels
