from viser.utils.utils import named_layers, read_image
from viser.attrs.smooth_grad import SmoothGrad
from flask import Flask,render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
from torch import nn
import torchvision
from PIL import Image
import numpy as np
from viser import ActivationsHook, FiltersHook, LayerHook
from viser.attrs import *
from viser.utils import *
import torchvision.transforms.functional as TF
import time
import matplotlib.cm as cm
import cv2

STATIC_FOLDER = 'static'

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder='static/web')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def hello_world():
    return render_template('index.html')
    
def get_input(filename: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    image = TF.to_tensor(Image.open(filename).convert('RGB'))
    return TF.normalize(image, mean, std).unsqueeze(0), image

images = { 
          'static/images/snake.jpg': 56, 
          'static/images/cat_dog.png' : 243,
          'static/images/spider.png': 72,
          'static/images/arctic_fox.jpeg': 279
}

@socketio.on('get_models')
def get_models():
    emit('models', torch_models())
    
@socketio.on('get_images')
def get_images():
    emit('images', images)

@socketio.on('get_layers')
def model_layers(data):
    model = get_model(data['model'])
    layers = []
    for index, (name, layer) in enumerate(named_layers(model)):
        layers.append({ 'index': index, 'name' : name, 'layer': str(layer) })
            
    emit('layers', layers)
            
@socketio.on('activations')
def handle_activations(data):    
    model = get_model(data['model'])
    x, _ = get_input(data['input'])
    
    activations_hook = ActivationsHook(model, stop_types=nn.Linear)

    model(x)

    emit('response_activations', activations_hook.save(f'static/out/alexnet_{time.time()}', normalization_scope='unit', split_channels=True))

@socketio.on('filters')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, _ = get_input(data['input'])
    
    filters_hook = FiltersHook(model)

    model(x)

    emit('response_filters', filters_hook.save(f'static/out/{data["model"]}_{time.time()}'))

@socketio.on('deep_dream')
def handle_guided_saliency(data):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(data['model'])
    model.to(device)
    model.eval()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    hook = LayerHook(model, int(data['layer']))
    
    image = Image.open(data['input'])
    x = TF.normalize(TF.resize(TF.to_tensor(image), [224, 224]), mean, std).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.SGD([x], lr=float(data['lr']), weight_decay=1e-4)
    
    for i in range(int(data['epochs'])):
        optimizer.zero_grad()
        model(x)
        loss = -torch.mean(hook.activations[0, int(data['activation'])])
        loss.backward()
        optimizer.step()
        
        filename = f'static/out/deep_dream_{i}.png'
        torchvision.utils.save_image(denormalize(x.detach(), mean, std, clamp=bool(data['clamp'])), filename, normalize=True)

        emit('response_deep_dream', {
            'epoch' : i,
            'loss' : loss.item(),
            'output': filename
        })

@socketio.on('class_max')
def handle_class_max(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    manual_seed(0)
    
    model = get_model(data['model'])
    model.to(device)
    model.eval()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    x = torch.randint(0, 255, [1, 3, 224, 224]) / 255
    x = TF.normalize(x, mean, std).to(device).requires_grad_(True)
    
    optimizer = torch.optim.SGD([x], lr=int(data['lr']), weight_decay=float(data['weight_decay']))
    
    for i in range(int(data['epochs'])):
        if bool(data['blur']) and i % int(data['blur_freq']) == 0:
            x.data = TF.gaussian_blur(x.data, [3, 3])
        
        optimizer.zero_grad()
        output = model(x)
        loss = -output[0, int(data['target'])]
        loss.backward()
        
        if bool(data['clip_grad']):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        
        optimizer.step()
        
        filename = f'static/out/class_max_{i}.png'
        torchvision.utils.save_image(denormalize(x.detach(), mean, std, clamp=bool(data['clamp'])), filename, normalize=True)

        emit('response_class_max', {
            'epoch' : i,
            'loss' : loss.item(),
            'output': filename
        })


@socketio.on('act_max')
def handle_act_max(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    manual_seed(0)
    
    model = get_model(data['model'])
    model.to(device)
    model.eval()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    hook = LayerHook(model, int(data['layer']))
    
    x = torch.randint(150, 180, [1, 3, 224, 224]) / 255
    x = TF.normalize(x, mean, std).to(device).requires_grad_(True)
    
    optimizer = torch.optim.Adam([x], lr=float(data['lr']), weight_decay=1e-6)
    
    for i in range(int(data['epochs'])):        
        optimizer.zero_grad()
        model(x)
        loss = -torch.mean(hook.activations[0, int(data['activation'])])
        loss.backward()    
        optimizer.step()
        
        filename = f'static/out/act_max_{i}.png'
        torchvision.utils.save_image(denormalize(x.detach(), mean, std, clamp=bool(data['clamp'])), filename, normalize=True)

        emit('response_act_max', {
            'epoch' : i,
            'loss' : loss.item(),
            'output': filename
        })

@socketio.on('saliency')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, original = get_input(data['input'])
    target = int(data['target'])

    saliency = Saliency(model)
    attributions = saliency.attribute(x, target, abs=False).squeeze(0)

    emit('response_saliency', {
        'vanilla Gradient' : save_image(attributions, f'static/out/grad_colorful_{time.time()}.png'),
        'Gradient(abs)' : save_image(torch.abs(attributions), f'static/out/grad_colorful_{time.time()}.png'),
        'Gradient(abs) * Image' : save_image(normalize(torch.abs(attributions)) * original, normalize=False, filename=f'static/out/grad_x_image_colorful_{time.time()}.png'),
        'Grayscale Gradient' : save_image(torch.sum(torch.abs(attributions), dim=0), f'static/out/grad_grayscale_{time.time()}.png'),
        'Grayscale Gradinet * Image' :save_image(torch.sum(torch.abs(attributions * x.squeeze(0).detach()), dim=0), filename=f'static/out/grad_x_image_{time.time()}.png')
    })

@socketio.on('guided_saliency')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, original = get_input(data['input'])
    target = int(data['target'])

    guided_saliency = GuidedSaliency(model)
    guided_attributions = guided_saliency.attribute(x, target, abs=False).squeeze(0)

    emit('response_guided_saliency', {
        'Guided Saliency' : save_image(guided_attributions, f'static/out/grad_colorful_{time.time()}.png'),
        'Guided Saliency(Abs)' : save_image(torch.abs(guided_attributions), f'static/out/grad_colorful_{time.time()}.png'),
        'Guided Saliency(Abs) * Image' : save_image(normalize(torch.abs(guided_attributions)) * original, normalize=False, filename=f'static/out/grad_colorful_{time.time()}.png'),
        'Guided Grayscale Saliency' : save_image(torch.sum(torch.abs(guided_attributions), dim=0), f'static/out/grad_grayscale_{time.time()}.png'),
        'Guided Grayscale Saliency * Image' :save_image(torch.sum(torch.abs(guided_attributions * x.squeeze(0).detach()), dim=0), f'static/out/grad_x_image_{time.time()}.png')
    })
    
@socketio.on('smooth_grad')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, _ = get_input(data['input'])
    noise_level = float(int(data['noise_level']) / 100)
    target = int(data['target'])

    smoothgrad = SmoothGrad(model)
    attributions = smoothgrad.attribute(x, noise_level=noise_level, target=target, epochs=int(data['epochs']), abs=False).squeeze(0)

    emit('response_smooth_grad', {
        'colorful' : save_image(attributions, f'static/out/grad_colorful_{time.time()}.png'),
        'grayscale' : save_image(torch.sum(torch.abs(attributions), dim=0), f'static/out/grad_grayscale_{time.time()}.png'),
        'grad_x_image' :save_image(torch.sum(torch.abs(attributions * x.squeeze(0).detach()), dim=0), f'static/out/grad_x_image_{time.time()}.png')
    })
    
@socketio.on('intergrated_grad')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, _ = get_input(data['input'])
    noise_level = float(int(data['noise_level']) / 100)
    target = int(data['target'])

    smoothgrad = SmoothGrad(model)
    attributions = smoothgrad.attribute(x, noise_level=noise_level, target=target, epochs=int(data['epochs']), abs=False).squeeze(0)

    emit('response_intergrated_grad', {
        'colorful' : save_image(attributions, f'static/out/grad_colorful_{time.time()}.png'),
        'grayscale' : save_image(torch.sum(torch.abs(attributions), dim=0), f'static/out/grad_grayscale_{time.time()}.png'),
        'grad_x_image' :save_image(torch.sum(torch.abs(attributions * x.squeeze(0).detach()), dim=0), f'static/out/grad_x_image_{time.time()}.png')
    })
    
@socketio.on('augmentedgrad')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, x_c = get_input(data['input'])
    target = int(data['target'])
    
    # mean = [0.485, 0.456, 0.406]
    # std  = [0.229, 0.224, 0.225]

    # original = Image.open(data['input']).convert('RGB')
    # opencvImage = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    # image = cv2.bilateralFilter(opencvImage,  19, 19 * 2, 19 / 2)
    # filename = f'static/out/bilateral_{time.time()}.png'
    # cv2.imwrite(filename, image)
    # emit('bilateral', { "image": filename })
    
    # x = TF.normalize(TF.to_tensor(image), mean, std).unsqueeze(0)
    # original = read_image(data['input'])
    
    ops = [
        # CroppedPad([224 - 16, 224], [16, 0]),
        # CroppedPad([224 - 16, 224], [0, 0]),
        # CroppedPad([224, 224 - 16], [0, 16]),
        # CroppedPad([224, 224 - 16], [0, 0]),
        # CroppedPad([224 - 16, 224 - 16], [16, 16]),
        # CroppedPad([224 - 16, 224 - 16], [0, 0]),
        Original(),
        # Invert(),
        HorizontalFlip(),
        # Compose([HorizontalFlip(), Invert()]),
        # VerticalFlip(),

        # GaussianBlur([3, 3]),
        GaussianBlur([5, 5]),
        GaussianBlur([7, 7]),
        GaussianBlur([9, 9]),
        # GaussianBlur([11, 11]),
        # GaussianBlur([13, 13]),
        # Compose([HorizontalFlip(), GaussianBlur([3, 3])]),
        Compose([HorizontalFlip(), GaussianBlur([5, 5])]),
        Compose([HorizontalFlip(), GaussianBlur([7, 7])]),
        Compose([HorizontalFlip(), GaussianBlur([9, 9])]),
        # Compose([HorizontalFlip(), GaussianBlur([11, 11])]),
        # Compose([HorizontalFlip(), GaussianBlur([13, 13])]),
        AdjustSharpness(1.35),
        AdjustSharpness(1.25),
        AdjustSharpness(1.15),
        AdjustSharpness(0.75),
        AdjustSharpness(0.85),
        AdjustSharpness(0.95),
        Compose([HorizontalFlip(), AdjustSharpness(0.95)]),
        Compose([HorizontalFlip(), AdjustSharpness(0.85)]),
        Compose([HorizontalFlip(), AdjustSharpness(0.75)]),
        Compose([HorizontalFlip(), AdjustSharpness(1.15)]),
        Compose([HorizontalFlip(), AdjustSharpness(1.25)]),
        Compose([HorizontalFlip(), AdjustSharpness(1.35)]),
        # AdjustBrightness(0.65),
        # AdjustBrightness(1.35),
        # AdjustContrast(0.65),
        # AdjustContrast(1.35),
        # Compose([HorizontalFlip(), AdjustBrightness(0.65)]),
        # Compose([HorizontalFlip(), AdjustBrightness(1.35)]),
        # Compose([HorizontalFlip(), AdjustContrast(0.65)]),
        # Compose([HorizontalFlip(), AdjustContrast(1.35)]),
        # Compose([VerticalFlip(), AdjustBrightness(0.75)]),
        # Compose([VerticalFlip(), AdjustBrightness(1.25)]),
        # Compose([VerticalFlip(), AdjustContrast(0.75)]),
        # Compose([VerticalFlip(), AdjustContrast(1.25)]),
        
        # AdjustHue(-0.1),
        # AdjustHue(0.1),
        # AdjustHue(-0.2),
        # AdjustHue(0.2),
        # AdjustHue(-0.05),
        # AdjustHue(0.05),
        # AdjustHue(-0.15),
        # AdjustHue(0.15),
        # Noise(20),
        # Compose([HorizontalFlip(), Noise(20)]),
        # Noise(2),
        # Noise(2),
        # Noise(2),
        # Noise(2),
    ]

    augmentgrad = AugmentedGrad(model)
    attributions = augmentgrad.attribute(x, ops, target, abs=False).squeeze(0)

    emit('response_augmentedgrad', {
        'Augmented Saliency' : save_image(attributions, f'static/out/grad_colorful_{time.time()}.png'),
        'Augmented Saliency(Abs)' : save_image(torch.abs(attributions), f'static/out/grad_colorful_{time.time()}.png'),
        'Augmented Saliency(Abs) * Image' :save_image(normalize(torch.abs(attributions)) * x_c, normalize=True, filename=f'static/out/grad_x_image_colorful_{time.time()}.png'),
        'Augmented Saliency(Grayscale)' : save_image(torch.sum(torch.abs(attributions), dim=0), f'static/out/grad_grayscale_{time.time()}.png'),
        'Augmented Saliency(Grayscale) * Image' :save_image(torch.sum(torch.abs(attributions * x.squeeze(0).detach()), dim=0),f'static/out/grad_x_image_{time.time()}.png')
    })
    
@socketio.on('gradcam')
def handle_saliency(data):    
    model = get_model(data['model'])
    x, image = get_input(data['input'])
    target = int(data['target'])

    gradcam = GradCAM(model, int(data['layer']))
    activations = gradcam.attribute(x, target).squeeze(0)
    
    saliency = GuidedSaliency(model)
    attributions = saliency.attribute(x, target, abs=False).squeeze(0)
    
    cam = normalize(activations)

    grad_cam = TF.to_pil_image(cam).resize([x.shape[2], x.shape[3]], resample=Image.ANTIALIAS)
    grayscale_filename = f'static/out/heatmap_grayscale_{time.time()}.png'
    grad_cam.save(grayscale_filename)

    cmap = cm.get_cmap('hsv')
    heatmap = cmap(TF.to_tensor(grad_cam)[0].detach().numpy())
    colorful_filename = f'static/out/heatmap_{time.time()}.png'
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(colorful_filename)

    heatmap[:, :, 3] = 0.4

    heatmap_on_image = Image.new('RGBA', (x.shape[2], x.shape[3]))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, TF.to_pil_image(image).convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, Image.fromarray((heatmap * 255).astype(np.uint8)))
    on_image_filename = f'static/out/heatmap_on_image_{time.time()}.png'
    heatmap_on_image.save(on_image_filename)
    
    # Vanilla gradients
    emit('response_gradcam', {
        'grayscale' : grayscale_filename,
        'colorful' : colorful_filename,
        'on_image' : on_image_filename,
        'guided_saliecy': save_image(attributions, f'static/out/guided_saliency_{time.time()}.png'),
        'guided_grad_cam': save_image(TF.to_tensor(grad_cam) * attributions, f'static/out/guided_gradcam_{time.time()}.png')
    })
    
@socketio.on_error_default  # Handles the default namespace
def error_handler(e):
    emit('error', { 'message': str(e) })

if __name__ == '__main__':
    socketio.run(app, debug=True)

