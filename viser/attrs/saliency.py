import torch
import torch.nn as nn

__all__ = ['Saliency']

class Saliency:
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''
        
        if not input.requires_grad:
            input.requires_grad_()
            
        if input.grad is not None:
            input.grad.zero_()

        output = self.model(input)
        loss = output[0, target] if target and target < output.shape[1] else output.max()
        
        grad = torch.autograd.grad(loss, input)[0]
        
        return torch.abs(grad) if abs else grad