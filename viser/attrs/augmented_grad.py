import torch
import torch.nn as nn
from .core import Attribution

__all__ = ['AugmentedGrad']


class AugmentedGrad(Attribution):
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model

    def attribute(self, input: torch.Tensor, ops: list, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''

        grads = []

        for op in ops:
            x = op(input)

            Attribution.prepare_input(x)

            output = self.model(x)
            loss = output[0, target] if target and target < output.shape[1] else output.max()

            grads.append(op.reverse(torch.autograd.grad(loss, x)[0]))

        grad = torch.cat(grads).mean(dim=0, keepdim=True)
        return torch.abs(grad) if abs else grad
