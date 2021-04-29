import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import normalize
from torchvision.transforms.transforms import Resize
from visor import ActivationsHook, FiltersHook
from PIL import Image

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')
    
    model = torchvision.models.alexnet(pretrained=True)
    model.to(device)

    activations_hook = ActivationsHook(model, stop_types=nn.Linear)
    filters_hook = FiltersHook(model)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(Image.open('images/cat_dog.png')).unsqueeze(0).to(device)

    model(image)

    activations = activations_hook.activations
    filters = filters_hook.filters
    
    json = activations_hook.save('logs/alexnet', normalization_scope='layer', split_channels=True)
    print(json)
    
    