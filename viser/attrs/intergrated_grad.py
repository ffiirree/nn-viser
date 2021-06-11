import torch
import torch.nn as nn
from .core import Attribution

__all__ = ['IntergratedGrad']

class IntergratedGrad(Attribution):
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, target: int = None, epochs: int=50, abs: bool = True):
        assert input.dim() == 4, ''
        
        Attribution.prepare_input(input)
        
        grad = torch.zeros(input.shape)

        for i in range(epochs):
            noise = torch.randn(input.shape) / 20.0
            input.data = input.data + noise.data
            output = self.model(input)
            loss = output[0, target] if target and target < output.shape[1] else output.max()
            
            grad += torch.autograd.grad(loss, input)[0]
        
        return torch.abs(grad / epochs) if abs else (grad / epochs)