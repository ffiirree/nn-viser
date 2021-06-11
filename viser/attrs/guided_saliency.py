import torch
import torch.nn as nn
from .core import Attribution

__all__ = ['GuidedSaliency']

class LayerHook(Attribution):
    def __init__(self, layer: nn.Module) -> None:
        self.activation = None
        
        layer.register_forward_hook(self.forward)
        layer.register_backward_hook(self.backward_hook)
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.activation = output

    def backward_hook(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        activation = self.activation[-1]
        activation[activation > 0] = 1
        modified_grad_output = activation * torch.clamp(grad_input[0], min=0.0)
        return (modified_grad_output, )

class GuidedSaliency:
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
        
        for layer in self.model.modules():
            if isinstance(layer, (nn.ReLU)):
                LayerHook(layer)
    
    def attribute(self, input: torch.Tensor, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''
        
        Attribution.prepare_input(input)

        output = self.model(input)
        loss = output[0, target] if target and target < output.shape[1] else output.max()
        
        grad = torch.autograd.grad(loss, input)[0]
        
        return torch.abs(grad) if abs else grad