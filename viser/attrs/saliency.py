import torch
import torch.nn as nn
from .core import Attribution

__all__ = ['Saliency']

class Saliency(Attribution):
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''
        
        Attribution.prepare_input(input)

        output = self.model(input)
        # loss = torch.sum(output, 1)
        loss = output[0, target] if target and target < output.shape[1] else output.max()
        
        grad = torch.autograd.grad(loss, input)[0]
        
        return torch.abs(grad) if abs else grad