import torch
import torch.nn as nn

class FGSM:
    def __init__(self, model) -> None:
        self.model = model
        self.model.eval()
    
    def generate(self, input: torch.Tensor, target: int = None, epsilon: float = 0.05, epochs: int = 1):
        assert input.dim() == 4, ''
        
        if not input.requires_grad:
            input.requires_grad_()

        if input.grad is not None:
            input.grad.zero_()

        output = torch.softmax(self.model(input), dim=1)
        loss = output[0, target]
        grad = torch.autograd.grad(loss, input)[0].sign()
        
        return grad, (input - epsilon * grad)