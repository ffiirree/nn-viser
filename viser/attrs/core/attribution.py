import abc
from typing import Callable
from torch import Tensor

__all__ = ['Attribution']


class Attribution(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def prepare_input(input: Tensor) -> None:
        if not input.requires_grad:
            input.requires_grad_()

        if input.grad is not None:
            input.grad.zero_()

    attribute: Callable
