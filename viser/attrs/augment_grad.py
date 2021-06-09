import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

__all__ = ['AugmentedGrad']


class AugmentedGrad:
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, ops: list, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''
            
        grad_sum = torch.zeros(input.shape)
        
        for op in ops:
            x = op(input)
            
            if not x.requires_grad:
                x.requires_grad_()
                
            if x.grad is not None:
                x.grad.zero_()
                
            output = self.model(x)
            loss = output[0, target] if target and target < output.shape[1] else output.max()
        
            grad = op.reverse(torch.autograd.grad(loss, x)[0])
            grad_sum += torch.abs(grad) if abs else grad
        
        return grad_sum / len(ops)