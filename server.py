from flask import Flask
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import torch
from torch import nn
import torchvision
from PIL import Image
import torchvision.transforms as T
import numpy as np
import io
import torch.nn.functional as F
import matplotlib.pyplot as plt
from visor import ActivationsHook

cmap = plt.get_cmap('RdBu')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def to_bwr(ts):
    assert ts[0].dim() == 2, f"dim must be 2, but is {ts[0].dim()} / {ts[0].shape}"

    low = min([l.min().item() for l in ts])
    high = max([ h.max().item() for h in ts])

    images = []
    for t in ts:
        image = torch.zeros([3, t.shape[0], t.shape[1]])
        norm_ip(t, low, high)
        tensor = t.mul(255).clamp_(0, 255).type(torch.uint8)

        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                color = cmap(tensor[i][j].item())
                
                image[0][i][j] = color[0] * 255
                image[1][i][j] = color[1] * 255
                image[2][i][j] = color[2] * 255
        images.append(image)
    return images

def tensor_to_bwr_file(t):
    print(t.shape)
    image = t.permute(1, 2, 0).cpu().detach().numpy()
    image = Image.fromarray(image.astype(np.uint8))
    file = io.BytesIO()
    image.save(file, 'PNG')
    file.seek(0)
    return file

def feature_maps_to_bwr(feature_maps):
    return [{ name: tensor_to_bwr_file(to_bwr([t.view([-1, t.shape[2]])])[0]).read() for name, t in group} for group in feature_maps]

@socketio.on('predict')
def predict(data):
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')

    model = torchvision.models.alexnet(pretrained=True)
    model.to(device)
    hook = ActivationsHook(model, stop_types=nn.Linear)

    # model = MnistNetTiny(11)
    # model.load_state_dict(torch.load(data['model']))
    # model.to(device=device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(Image.open('cat_dog.png')).unsqueeze(0).to(device)

    # hook = FeatureMapsHook(model)
    model(image)

    # feature_maps = feature_maps_to_bwr(hook.feature_maps)
    
    
    emit('net', hook.save('static/alexnet', split_channels=False))

if __name__ == '__main__':
    socketio.run(app, debug=True)

