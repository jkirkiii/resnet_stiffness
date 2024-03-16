import argparse
import pathlib
import time
import torch
import torchvision

from resnet import StiffnessLoss, resnet20

from learners import test, train
from utils import format_time, logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lagrange', default=.1, type=float)
    parser.add_argument('--learning-rate', default=.1, type=float, help='initial learning rate')
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('--momentum', default=.9, type=float, help='momentum (default: .9)')
    parser.add_argument('--ours', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('--weight-decay', default=1.e-4, type=float, help='weight decay (default: 1.e-4)')
    arguments = parser.parse_args()

    train_transforms = [torchvision.transforms.ToTensor()]
    test_transforms = [torchvision.transforms.ToTensor()]

    train_transforms = [
        torchvision.transforms.RandomCrop(32, padding=4, pad_if_needed=True, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip(),
    ] + train_transforms

    normalization = torchvision.transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
    train_transforms += [normalization]
    test_transforms += [normalization]

    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.Compose(train_transforms), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.Compose(test_transforms), download=True)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = None

    if arguments.model == 'resnet20': model = resnet20()
    else: pass

    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=arguments.learning_rate, momentum=arguments.momentum, weight_decay=arguments.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    best = {'train_loss': float('inf'), 'train_epoch': 0, 'test_loss': float('inf'), 'test_epoch': 0}
    save_path = pathlib.Path(arguments.model)
    checkpoints_path = save_path / 'checkpoints'
    save_path.mkdir(exist_ok=True)
    checkpoints_path.mkdir(exist_ok=True)

    our_loss = StiffnessLoss(arguments.lagrange, arguments.batch_size) if arguments.ours else None

    start_time = time.time()

    for epoch in range(1, arguments.epochs+1):
        epoch_time = time.time()
        train_accuracy, train_loss, stiffness_loss = train(model, train_data, device, objective, optimizer, our_loss)
        test_accuracy, test_loss = test(model, test_data, device, objective)
        scheduler.step(train_loss)
        if train_loss < best['train_loss']: best['train_loss'], best['train_epoch'] = train_loss, epoch
        if test_loss < best['test_loss']: best['test_loss'], best['test_epoch']= test_loss, epoch
        torch.save(model.state_dict(), checkpoints_path / f'{epoch:03}.pt')
        end_time = time.time()

        logger([
            epoch, stiffness_loss,
            train_accuracy, train_loss, best['train_epoch'], test_accuracy, test_loss, best['test_epoch'],
            format_time(end_time - start_time),
            format_time((end_time - start_time) * (arguments.epochs - epoch) / epoch),
            format_time(end_time - epoch_time),
            *(scheduler.get_last_lr() if scheduler != None else []),
        ], save_path)

if __name__ == '__main__': main()
