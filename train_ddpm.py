from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time
import argparse
import functools
import torchsummary
from EfficientLayers import CustomConv2d, CustomLinear
from utils import train_ddpm, test_ddpm, DropSchedular
from torch.utils.tensorboard import SummaryWriter
from models import DDPM


tasks = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CelebA': datasets.CelebA
}
def crop_celeba(img):
    return transforms.functional.crop(img, top=40, left=15, height=148, width=148)

# normalize to [-1, 1]
data_transforms = {
    'MNIST': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]),
    'FashionMNIST': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]),
    'CelebA': transforms.Compose([
        crop_celeba,
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

def createDDPM(*args, **kwargs):
    return DDPM(*args, **kwargs)

models = {
    'ddpm': createDDPM
}


# CUDA_VISIBLE_DEVICES=1 python train_ddpm.py --exp runs/generation --task MNIST --mode normal --model ddpm --drop_mode constant --epochs 300 --lr 0.001 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0
# CUDA_VISIBLE_DEVICES=2 python train_ddpm.py --exp runs/generation --task MNIST --mode efficient --model ddpm --drop_mode constant --epochs 300 --lr 0.001 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0

# CUDA_VISIBLE_DEVICES=0 python train_ddpm.py --exp runs/generation --task FashionMNIST --mode normal --model ddpm --drop_mode constant --epochs 500 --lr 0.001 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0
# CUDA_VISIBLE_DEVICES=3 python train_ddpm.py --exp runs/generation --task FashionMNIST --mode efficient --model ddpm --drop_mode constant --epochs 500 --lr 0.001 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0

# CUDA_VISIBLE_DEVICES=1 python train_ddpm.py --exp runs/generation --task CelebA --mode normal --model ddpm --drop_mode constant --epochs 200 --lr 0.0002 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0
# CUDA_VISIBLE_DEVICES=2 python train_ddpm.py --exp runs/generation --task CelebA --mode efficient --model ddpm --drop_mode constant --epochs 200 --lr 0.0002 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TauPET inference")
    parser.add_argument("--task", type=str, default='MNIST', help="MNIST, FashionMNIST, CelebA")
    parser.add_argument("--mode", type=str, default='normal', help="normal, efficient mode")
    parser.add_argument("--model", type=str, default='ddpm', help="ddpm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--unified", action='store_true', help="Unified pruning")
    parser.add_argument("--builtin", action='store_true', help="Use built-in conv2d")
    parser.add_argument("--drop_mode", type=str, default='cosine', help="Drop mode: constant, linear, cosine")
    parser.add_argument("--percentage", type=float, default=0.946, help="Percentage of filters to be pruned")
    parser.add_argument("--min_percentage", type=float, default=0.2, help="Minimum percentage of filters to be pruned")
    parser.add_argument("--interleave", action='store_true', help="Interleave pruning")
    parser.add_argument("--warmup", type=int, default=0, help="Number of epochs for warmup (normal training for a while)")
    parser.add_argument("--exp", type=str, default='runs', help="Experiment name")
    parser.add_argument("--schedule", type=str, default='linear', help="Schedule for beta")
    parser.add_argument("--subsample", action='store_true', help="Subsample CelebA dataset")
    args = parser.parse_args()

    if args.unified == False:
        print('Warning: Split mode is deprecated. Auto setting to unified mode.')
    args.unified = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = args.task
    input_channel = 3 if task == 'CelebA' else 1
    img_size = 64 if task == 'CelebA' else 32

    if task != 'CelebA':
        timesteps = 200
        model_channels = 16
        attention_resolutions = [4,2,1]
        channel_mult = (1, 2, 4)
        num_attention_blocks = [0, 0, 2]

        train_dataset = tasks[task](root='./data', train=True, transform=data_transforms[task], download=True)
        # split the training dataset into training and validation
        train_size, val_size = int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False)

        test_dataset = tasks[task](root='./data', train=False, transform=data_transforms[task], download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False)
    else:
        timesteps = 1000
        model_channels = 128
        attention_resolutions = [8,4,2,1]
        channel_mult = (1, 2, 2, 2)
        num_attention_blocks = [0, 0, 0, 2]

        train_dataset = datasets.CelebA(root='./data', split='train', transform=data_transforms['CelebA'], download=True)
        valid_dataset = datasets.CelebA(root='./data', split='valid', transform=data_transforms['CelebA'], download=True)
        test_dataset = datasets.CelebA(root='./data', split='test', transform=data_transforms['CelebA'], download=True)
        if args.subsample:
            # only take 1/4 from each dataset
            train_dataset = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset), 4))
            valid_dataset = torch.utils.data.Subset(valid_dataset, range(0, len(valid_dataset), 4))
            test_dataset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset), 4))
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8)
        val_loader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=8)

    if args.mode == 'efficient':
        save_dir = f'{args.exp}/{args.task}/{args.model}/{args.seed}_{args.mode}_bs{args.bs}_lr{args.lr}_{args.drop_mode}_{args.percentage}_{args.min_percentage}_{args.schedule}_step{timesteps}_{"no-" if not args.interleave else ""}inter_warmup{args.warmup}' + ('_smallCelebA' if task == 'CelebA' and args.subsample else '')
    else:
        save_dir = f'{args.exp}/{args.task}/{args.model}/{args.seed}_{args.mode}_bs{args.bs}_lr{args.lr}_{args.schedule}_step{timesteps}' + ('_smallCelebA' if task == 'CelebA' and args.subsample else '')

    mode = args.mode
    nn.Conv2d = functools.partial(CustomConv2d, mode=mode, percentage=args.percentage, unified=args.unified, built_in=args.builtin, device=device)

    model = models[args.model](
        timesteps=timesteps,
        beta_schedule=args.schedule, 
        loss_type='l2',
        image_size=img_size,
        channels=input_channel,
        model_channels=model_channels,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        num_attention_blocks=num_attention_blocks,
        device=device
    ).to(device)
    result, _ = torchsummary.summary(model, (input_channel, img_size, img_size))

    # save the model summary into txt file
    with open(f'/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/KernelConv/model_summary/{save_dir.replace("/", "_")}.txt', 'w') as f:
        f.write(result)
    
    exit()

    dropschedular = DropSchedular(model, args.drop_mode, args.percentage, args.min_percentage, args.epochs, interleave=args.interleave, warmup=args.warmup)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    writer = SummaryWriter(save_dir)

    best_recon_loss = float('inf')

    start = time.time()
    for epoch in range(args.epochs):
        s = time.time()
        train_ddpm(model, device, train_loader, dropschedular, optimizer, epoch, writer)
        print(f'Time taken for epoch: {time.time() - s:.2f} seconds')
        writer.add_scalar('training time / epoch', time.time() - s, epoch)

        # if (epoch + 1) % 1 == 0:
        val_loss = test_ddpm(model, device, val_loader, epoch, writer, "Validation")
        if val_loss < best_recon_loss:
            best_recon_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')

            test_ddpm(model, device, test_loader, epoch, writer)

    print(f'Total time taken: {time.time() - start:.2f} seconds')

    # save the model
    torch.save(model.state_dict(), f'{save_dir}/latest_model.pth')

    # kernelsummary.kernelsummary(model, next(iter(train_loader))[0].to(device), savedir='./kernelsummary/CIFAR100')