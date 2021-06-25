from .core import Attribution
import torch
import torch.nn as nn

__all__ = ['SmoothGrad']

class SmoothGrad(Attribution):
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, noise_level: float = 0.1, target: int = None, epochs: int = 50, abs: bool = True):
        assert input.dim() == 4, ''
        
        grads = []
        std = noise_level * (input.max() - input.min())
        print(noise_level)
        
        for _ in range(epochs):
            x = input.detach().clone() + torch.normal(mean=0, std=std, size=input.shape)

            Attribution.prepare_input(x)
                
            output = self.model(x)
            loss = output[0, target] if target and target < output.shape[1] else output.max()
            
            # loss = torch.sum(output, dim=1)
            
            grads.append(torch.autograd.grad(loss, x)[0])
        
        grad = torch.cat(grads).mean(dim=0, keepdim=True)
        return torch.abs(grad) if abs else grad