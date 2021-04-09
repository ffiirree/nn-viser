from flask import Flask
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import torch
from models import MnistNetTiny
from PIL import Image
import torchvision.transforms.functional as T
import numpy as np
import io
import torch.nn.functional as F
import matplotlib.pyplot as plt

cmap = plt.get_cmap('RdBu')

app = Flask(__name__)
CORS(app, cors_allowed_origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def to_bwr(ts):
    assert ts[0].dim() == 2, f"dim must be 2, but is {ts[0].dim()}"

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
    image = t.cpu().permute(1, 2, 0).detach().numpy()
    image = Image.fromarray(image.astype(np.uint8))
    file = io.BytesIO()
    image.save(file, 'PNG')
    file.seek(0)
    return file

@socketio.on('predict')
def predict(data):

    model = MnistNetTiny(11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(data['model']))
    model.to(device=device)

    x = Image.open(data['input'])
    x = T.to_tensor(x).to(device)

    conv1 = model.conv1(x.unsqueeze(0))
    relu1 = F.relu(conv1)
    conv2 = model.conv2(relu1)
    relu2 = F.relu(conv2)
    conv3 = model.conv3(relu2)
    relu3 = F.relu(conv3)
    conv4 = model.conv4(relu3)
    relu4 = F.relu(conv4)
    conv5 = model.conv5(relu4)
    relu5 = F.relu(conv5)

    avg = model.avg(relu5).squeeze()
    print(F.softmax(avg))

    input = to_bwr([x.squeeze().cpu()])[0]

    conv1 = conv1.squeeze().cpu()
    conv1 = conv1.view([-1, conv1.shape[1]])
    
    relu1 = relu1.squeeze().cpu()
    relu1 = relu1.view([-1, relu1.shape[1]])

    conv1_image, relu1_image = to_bwr([conv1, relu1])
    
    conv2 = conv2.squeeze().cpu()
    conv2 = conv2.view([-1, conv2.shape[1]])
    
    relu2 = relu2.squeeze().cpu()
    relu2 = relu2.view([-1, relu2.shape[1]])

    conv2_image, relu2_image = to_bwr([conv2, relu2])

    conv3 = conv3.squeeze().cpu()
    conv3 = conv3.view([-1, conv3.shape[1]])
    
    relu3 = relu3.squeeze().cpu()
    relu3 = relu3.view([-1, relu3.shape[1]])

    conv3_image, relu3_image = to_bwr([conv3, relu3])

    conv4 = conv4.squeeze().cpu()
    conv4 = conv4.view([-1, conv4.shape[1]])
    
    relu4 = relu4.squeeze().cpu()
    relu4 = relu4.view([-1, relu4.shape[1]])

    conv4_image, relu4_image = to_bwr([conv4, relu4])

    conv5 = conv5.squeeze().cpu()
    conv5 = conv5.view([-1, conv5.shape[1]])
    
    relu5 = relu5.squeeze().cpu()
    relu5 = relu5.view([-1, relu5.shape[1]])

    conv5_image, relu5_image = to_bwr([conv5, relu5])
    
    emit('net', { 
        'input' : tensor_to_bwr_file(input).read(),

        'conv1' : tensor_to_bwr_file(conv1_image).read(),
        'relu1' : tensor_to_bwr_file(relu1_image).read(),

        'conv2' : tensor_to_bwr_file(conv2_image).read(),
        'relu2' : tensor_to_bwr_file(relu2_image).read(),

        'conv3' : tensor_to_bwr_file(conv3_image).read(),
        'relu3' : tensor_to_bwr_file(relu3_image).read(),

        'conv4' : tensor_to_bwr_file(conv4_image).read(),
        'relu4' : tensor_to_bwr_file(relu4_image).read(),

        'conv5' : tensor_to_bwr_file(conv5_image).read(),
        'relu5' : tensor_to_bwr_file(relu5_image).read()
        })

if __name__ == '__main__':
    socketio.run(app, debug=True)

