import argparse
import torch
import torchvision
import torchvision.transforms.functional as TF
from visor import *

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE'
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-class',   type=int,   default=130)
    parser.add_argument('--epochs',         type=int,   default=225)
    parser.add_argument('--lr',             type=float, default=3)
    parser.add_argument('--clamp',          default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--blur',           default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--blur-freq',      type=int, default=2)
    parser.add_argument('--weight-decay',   type=float, default=0)
    parser.add_argument('--clip-grad',      default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--output-dir',     default='logs')
    args = parser.parse_args()

    print(args)

    manual_seed(0)

    model = torchvision.models.alexnet(pretrained=True)
    model.to(device)
    model.eval()

    x = torch.randint(0, 255, [1, 3, 224, 224]) / 255
    x = TF.normalize(x, mean, std).to(device).requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.epochs):
        if args.blur and i % args.blur_freq == 0:
            x.data = TF.gaussian_blur(x.data, [3, 3])
        
        optimizer.zero_grad()
        output = model(x)
        loss = -output[0, args.target_class]
        print(f'Iter: {i:>3}: loss = {loss.item()}')
        loss.backward()

        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        torchvision.utils.save_image(denormalize(x.detach(), mean, std, clamp=args.clamp), f'{args.output_dir}/generate_class_{args.target_class}_blur.png', normalize=True)
