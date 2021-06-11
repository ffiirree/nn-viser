import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from viser.hooks import LayerHook
from .core import Attribution

__all__ = ['GradCAM']

class GradCAM(Attribution):
    def __init__(self, model: Module, layer_index: int) -> None:
        self.model = model
        self.layer_index = layer_index
        self.hook = LayerHook(self.model, self.layer_index)
        
        self.model.eval()
            
    def attribute(self, input: Tensor, target: int = None, relu_attributions: bool = False):
        assert input.dim() == 4, ""
        
        Attribution.prepare_input(input)

        output = self.model(input)
        loss = output[0, target] if target and target < output.shape[1] else output.max()
        
        activations = self.hook.activations
        gradients = torch.autograd.grad(loss, activations)[0]
        
        summed_grads = torch.mean(gradients, (2, 3), keepdim=True)
        scaled_activations = torch.sum(summed_grads * activations, dim=1, keepdim=True)
        
        return scaled_activations if not relu_attributions else torch.relu(relu_attributions)