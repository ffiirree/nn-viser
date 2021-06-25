import torch
import torch.nn as nn
from .core import Attribution

__all__ = ['RelativeGrad']

class RelativeGrad(Attribution):
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''
        
        Attribution.prepare_input(input)

        output = self.model(input)
        
        loss = output[0, target] if target and target < output.shape[1] else output.max()
        # loss = torch.sum(output, 1)
        grad = torch.autograd.grad(loss, input)[0]
        
        x = input.detach().clone()
        Attribution.prepare_input(x)
        output = self.model(x)
        output[0, target] = 0
        loss = torch.sum(output, 1)
        grad_all = torch.autograd.grad(loss, x)[0]
        
        return torch.abs(grad_all) if abs else (grad - grad_all)