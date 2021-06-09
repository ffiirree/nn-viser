import torch
import torch.nn as nn

__all__ = ['IntergratedGrad']

class IntergratedGrad:
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, target: int = None, epochs: int=50, abs: bool = True):
        assert input.dim() == 4, ''
        
        if not input.requires_grad:
            input.requires_grad_()
            
        if input.grad is not None:
            input.grad.zero_()
        
        grad = torch.zeros(input.shape)

        for i in range(epochs):
            noise = torch.randn(input.shape) / 20.0
            input.data = input.data + noise.data
            output = self.model(input)
            loss = output[0, target] if target and target < output.shape[1] else output.max()
            
            grad += torch.autograd.grad(loss, input)[0]
        
        return torch.abs(grad / epochs) if abs else (grad / epochs)