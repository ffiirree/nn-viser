from torch import nn

__all__ = ['FeatureMapsHook']

class FeatureMapsHook:
    def __init__(self, model) -> None:
        self.feature_maps = []
        self.index = 0

        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.ReLU)):
                layer.register_forward_hook(self)

    def __call__(self, module, input, output):
        if isinstance(module, nn.Conv2d):
            self.feature_maps.append([])
            self.index = len(self.feature_maps) - 1
            self.feature_maps[self.index].append((f'conv2d_{self.index}', output.detach().clone().squeeze()))
        elif isinstance(module, nn.ReLU):
            self.feature_maps[self.index].append((f'relu_{self.index}', output.detach().clone().squeeze()))
        else:
            raise ValueError

    def clear(self):
        self.feature_maps.clear()
        self.index = 0

    def __str__(self) -> str:
        return [[(name, list(t.shape)) for name, t in group] for group in self.feature_maps].__str__()