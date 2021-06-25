import torch
import torch.nn as nn
from .core import Attribution
from flask_socketio import SocketIO, emit
__all__ = ['AccumulatedGrad']

class AccumulatedGrad(Attribution):
    def __init__(self, model: nn.Module) -> None:
        model.eval()
        self.model = model
    
    def attribute(self, input: torch.Tensor, epochs:int = 5, target: int = None, abs: bool = True):
        assert input.dim() == 4, ''
        
        grads = []
        for i in range(epochs):
            Attribution.prepare_input(input)

            output = self.model(input)
            topk1 = output.topk(5, 1, True, True)
            emit('log', { 'data' : f'[#{i}] {topk1.values.detach().numpy()} {topk1.indices.detach().numpy()}' })

            topk = torch.softmax(output, dim=1).topk(5, 1, True, True)
            emit('log', { 'data' : f'[%{i}] {(topk.values * 100).detach().numpy()} {topk.indices.detach().numpy()}' })
            loss = output[0, target] if target and target < output.shape[1] else output.max()
            print(f'output: {output[0, target]}, soft: {torch.softmax(output, dim=1)[0, target]}')
            
            grad = torch.autograd.grad(loss, input)[0]
            grads.append(grad * output[0, target])
            
            # print(input.shape, grad.shape)
            
            input.data -= grad.data
            print(f'input: {input.min()}, {input.max()}')
            print(f'grad: {grad.min()}, {grad.max()}')
            
            # emit('log', { 'data': 'xxx' })
        
        # grad = torch.cat(grads).mean(dim=0, keepdim=True)
        return torch.abs(grad) if abs else grad, input