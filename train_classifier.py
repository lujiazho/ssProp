import os
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
from utils import train, test, DropSchedular, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from models import ResNet6, ResNet18, ResNet26, ResNet34, ResNet50, MLP_Nlayer, CNN_Nlayer


tasks = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
    'CelebA': datasets.CelebA,
    'Flowers102': datasets.Flowers102,
    'ImageNet': datasets.ImageNet
}

transforms = {
    'MNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'FashionMNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ]),
    'CIFAR10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'CIFAR100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
    'CelebA': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'Flowers102': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # mean and std for ImageNet
    ]),
    'ImageNet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # mean and std for ImageNet
    ])
}

models = {
    'mlp': MLP_Nlayer,
    'cnn': CNN_Nlayer,
    'resnet6': ResNet6,
    'resnet18': ResNet18,
    'resnet26': ResNet26,
    'resnet34': ResNet34,
    'resnet50': ResNet50
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TauPET inference")
    parser.add_argument("--task", type=str, default='CIFAR100', help="MNIST, FashionMNIST, CIFAR10, CIFAR100")
    parser.add_argument("--mode", type=str, default='normal', help="normal, efficient mode")
    parser.add_argument("--model", type=str, default='resnet18', help="resnet18, resnet50, mlp, cnn")
    parser.add_argument("--nlayer", type=int, default=3, help="Number of layers only for simple MLP/CNN")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--unified", action='store_true', help="Unified pruning")
    parser.add_argument("--sparse", action='store_true', help="Sparse mode")
    parser.add_argument("--builtin", action='store_true', help="Use built-in conv2d")
    parser.add_argument("--random", action='store_true', help="Random pruning")
    parser.add_argument("--only_W", action='store_true', help="Only prune W")
    parser.add_argument("--drop_mode", type=str, default='cosine', help="Drop mode: constant, linear, cosine")
    parser.add_argument("--percentage", type=float, default=0.946, help="Percentage of filters to be pruned")
    parser.add_argument("--min_percentage", type=float, default=0.2, help="Minimum percentage of filters to be pruned")
    parser.add_argument("--interleave", action='store_true', help="Interleave pruning")
    parser.add_argument("--warmup", type=int, default=0, help="Number of epochs for warmup (normal training for a while)")
    parser.add_argument("--by_epoch", action='store_true', help="Prune by epoch")
    parser.add_argument("--T", type=int, default=10, help="Number of iterations for by iteration pruning")
    parser.add_argument("--exp", type=str, default='runs', help="Experiment name")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--resume", action='store_true', help="Resume training")
    # parser.add_argument("--resume_log", type=str, default=None, help="Resume training log")
    args = parser.parse_args()

    if args.unified == False:
        print('Warning: Split mode is deprecated. Auto setting to unified mode.')
    args.unified = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode == 'efficient':
        save_dir = f'{args.exp}/{args.task}/{args.model}/{args.seed}_{args.mode if not args.random else "random"}_bs{args.bs}_lr{args.lr}{"_sparse" if args.sparse else ""}_{args.drop_mode}_{args.percentage}_{args.min_percentage}_{"no-" if not args.interleave else ""}inter_warmup{args.warmup}' + (f"_byiter{args.T}" if not args.by_epoch else "") + (f"_onlyW" if args.only_W else "") + (f'_{args.nlayer}layer' if args.model == 'mlp' or args.model == 'cnn' else '') + (f'_dropout{args.dropout}' if args.dropout > 0 else '')
    else:
        save_dir = f'{args.exp}/{args.task}/{args.model}/{args.seed}_{args.mode}_bs{args.bs}_lr{args.lr}' + (f'_{args.nlayer}layer' if args.model == 'mlp' or args.model == 'cnn' else '') + (f'_dropout{args.dropout}' if args.dropout > 0 else '')
    # if os.path.exists(save_dir + '/latest_model.pth'):
    #     print(f"Model already exists in {save_dir}! Skipping...")
    #     exit()
    print(f'Saving to {save_dir}')
    
    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    task = args.task
    if task == 'ImageNet':
        num_classes = 1000
        input_channel = 3
        img_size = 224
    elif task == 'Flowers102':
        num_classes = 102
        input_channel = 3
        img_size = 128
    elif task == 'CelebA':
        num_classes = 40  # CelebA has 40 attribute labels
        input_channel = 3
        img_size = 64
    elif task == 'CIFAR100':
        num_classes = 100
        input_channel = 3
        img_size = 32
    elif task == 'CIFAR10':
        num_classes = 10
        input_channel = 3
        img_size = 32
    elif task == 'MNIST' or task == 'FashionMNIST':
        num_classes = 10
        input_channel = 1
        img_size = 28
    else:
        raise ValueError(f"Unknown task: {task}")

    if task == 'CelebA':
        train_dataset = datasets.CelebA(root='./data', split='train', transform=transforms['CelebA'], download=True)
        valid_dataset = datasets.CelebA(root='./data', split='valid', transform=transforms['CelebA'], download=True)
        test_dataset = datasets.CelebA(root='./data', split='test', transform=transforms['CelebA'], download=True)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8)
        val_loader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
    elif task == 'Flowers102':
        train_dataset = datasets.Flowers102(root='./data', split='train', transform=transforms['Flowers102'], download=True)
        val_dataset = datasets.Flowers102(root='./data', split='val', transform=transforms['Flowers102'], download=True)
        test_dataset = datasets.Flowers102(root='./data', split='test', transform=transforms['Flowers102'], download=True)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
    elif task == 'ImageNet':
        dataset = datasets.ImageNet(root='./data', split='val', transform=transforms['ImageNet'])

        # Split validation dataset into validation and test sets
        val_size = int(0.8 * len(dataset))
        test_size = len(dataset) - val_size
        val_dataset, test_dataset = torch.utils.data.random_split(dataset, [val_size, test_size])

        train_dataset = datasets.ImageNet(root='./data', split='train', transform=transforms['ImageNet'])
        # print(dataset.class_to_idx == train_dataset.class_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
    else:
        train_dataset = tasks[task](root='./data', train=True, transform=transforms[task], download=True)
        # split the training dataset into training and validation
        train_size, val_size = int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False)

        test_dataset = tasks[task](root='./data', train=False, transform=transforms[task], download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False)

    mode = args.mode
    drop_behavior = "" if "sensitivity_1" not in save_dir else args.exp.split('/')[-1] # only used while do sensitivity 1 analysis

    nn.Conv2d = functools.partial(CustomConv2d, mode=mode, percentage=args.percentage, unified=args.unified, sparse=args.sparse, built_in=args.builtin, random=args.random, only_W=args.only_W, drop_behavior=drop_behavior, device=device)
    if args.model == 'mlp':
        nn.Linear = functools.partial(CustomLinear, mode=mode, percentage=args.percentage, unified=args.unified, device=device)

    if args.model == 'mlp':
        model = models[args.model](input_size=img_size*img_size*input_channel, hidden_size=256, output_size=num_classes, num_layers=args.nlayer).to(device)
        result, _ = torchsummary.summary(model, (img_size*img_size*input_channel,), device=device)
    elif args.model == 'cnn':
        model = models[args.model](num_classes=num_classes, input_channel=input_channel, n_layer=args.nlayer).to(device)
        result, _ = torchsummary.summary(model, (input_channel, img_size, img_size), device=device)
    else:
        model = models[args.model](num_classes=num_classes, input_channel=input_channel, image_size=img_size, dropout=args.dropout).to(device)
        result, _ = torchsummary.summary(model, (input_channel, img_size, img_size), device=device)
    
    best_acc = 0
    start_epoch = 0
    dropschedular = DropSchedular(model, args.drop_mode, args.percentage, args.min_percentage, args.epochs, interleave=args.interleave, warmup=args.warmup, by_epoch=args.by_epoch, T=args.T)

    # # save the model summary into txt file
    # with open(f'/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/KernelConv/model_summary/{save_dir.replace("/", "_")}.txt', 'w') as f:
    #     f.write(result)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.resume:
    # if args.resume_path is not None and args.resume_log is not None:
        # model.load_state_dict(torch.load(args.resume_path))
        # # read the epoch number from the tensorboard log file
        # from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        # event_acc = EventAccumulator(args.resume_log)
        # event_acc.Reload()
        # start_epoch = event_acc.Scalars('test accuracy')[-1].step + 1
        assert os.path.exists(save_dir + '/best_checkpoint.pth'), f"Checkpoint not found in {save_dir}."
        start_epoch = load_checkpoint(model, optimizer, save_dir, "best_checkpoint")
        
        # this is only meaningful when by_epoch and interleave
        dropschedular.last_time = False if start_epoch % 2 == 0 else True

        start_epoch += 1
    
    if task == 'CelebA':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(save_dir)

    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        s = time.time()
        train(model, task, device, train_loader, dropschedular, optimizer, criterion, epoch, writer, args.model)
        print(f'Time taken for epoch: {time.time() - s:.2f} seconds')
        writer.add_scalar('training time / epoch', time.time() - s, epoch)

        # if (epoch + 1) % 1 == 0:
        val_acc = test(model, task, device, val_loader, criterion, epoch, writer, "Validation")
        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            save_checkpoint(model, optimizer, epoch, save_dir, "best_checkpoint")

            test(model, task, device, test_loader, criterion, epoch, writer)

    print(f'Total time taken: {time.time() - start:.2f} seconds')

    # # save the model
    # torch.save(model.state_dict(), f'{save_dir}/latest_model.pth')
    save_checkpoint(model, optimizer, epoch, save_dir, "latest_checkpoint")

    # kernelsummary.kernelsummary(model, next(iter(train_loader))[0].to(device), savedir='./kernelsummary/CIFAR100')