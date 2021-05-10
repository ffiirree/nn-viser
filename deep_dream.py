import argparse
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from visor import *

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',         type=int,   default=251)
    parser.add_argument('--layer',          type=int,   default=34)
    parser.add_argument('--activation',     type=int,   default=94)
    parser.add_argument('--lr',             type=float, default=12)
    parser.add_argument('--clamp',          default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--output-dir',     default='logs')
    args = parser.parse_args()

    print(args)

    model = torchvision.models.vgg19(pretrained=True)
    model.to(device)

    hook = LayerForwardHook(model, args.layer)

    image = Image.open('images/dd_tree.jpg')
    x = TF.normalize(TF.resize(TF.to_tensor(image), [224, 224]), mean, std).to(device).unsqueeze(0).requires_grad_(True)

    print(x.shape)
    optimizer = torch.optim.SGD([x], lr=args.lr, weight_decay=1e-4)
    model.eval()

    for i in range(args.epochs):
        optimizer.zero_grad()

        model(x)
        
        loss = -torch.mean(hook.activations[0, args.activation])
        print(f'Iter: {i:>2}: loss = {loss.item()}')
        loss.backward()
        optimizer.step()

        torchvision.utils.save_image(
            denormalize(x.detach(), mean, std, clamp=args.clamp), 
            f'{args.output_dir}/deep_dream_{args.layer}_{args.activation}.png', 
            normalize=True
        )

