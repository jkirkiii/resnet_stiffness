import argparse
import datetime
import pathlib
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from resnet import resnet20

from data import cifar10
from learners import test, train
from utils import format_time, logger
from stiffness import TotalNeuralStiffness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument('--lagrange', default=1., type=float)
    parser.add_argument('--learning-rate', default=.1, type=float, help='initial learning rate')
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('--momentum', default=.9, type=float, help='momentum (default: .9)')
    parser.add_argument('--ours', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('--weight-decay', default=5.e-4, type=float, help='weight decay (default: 1.e-4)')
    parser.add_argument('--spectral-norm', action=argparse.BooleanOptionalAction, default=True, type=bool)
    parser.add_argument('--num-divisions', default=1, type=int)
    parser.add_argument('--division', default=1, type=int)
    arguments = parser.parse_args()

    model = None

    if arguments.model == 'resnet20': model = resnet20(arguments.spectral_norm)
    else: pass

    train_data, test_data = cifar10(arguments.batch_size, arguments.num_divisions, arguments.division)
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=arguments.learning_rate, momentum=arguments.momentum, weight_decay=arguments.weight_decay)
    regularizer = TotalNeuralStiffness(lagrange=arguments.lagrange) if arguments.ours and arguments.lagrange > 0 else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=10)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if arguments.device == '' else arguments.device
    model.to(device)

    save_path = pathlib.Path('.') / 'experiments' / f'{arguments.model}-{datetime.datetime.now().strftime("%y%m%d%H%M%S")}'
    checkpoints_path = save_path / 'checkpoints'
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path.mkdir(exist_ok=True)
    print('make', save_path)

    best = {'train_loss': float('inf'), 'train_epoch': 0, 'test_loss': float('inf'), 'test_epoch': 0}
    start_time = time.time()

    for epoch in range(1, arguments.epochs+1):
        epoch_time = time.time()
        train_accuracy, train_loss, train_regularization = train(model, train_data, device, objective, optimizer, regularizer)
        test_accuracy, test_loss, test_regularization = test(model, test_data, device, objective, regularizer)
        scheduler.step(train_loss)
        if train_loss < best['train_loss']: best['train_loss'], best['train_epoch'] = train_loss, epoch
        if test_loss < best['test_loss']: best['test_loss'], best['test_epoch']= test_loss, epoch
        torch.save(model.state_dict(), checkpoints_path / f'{epoch:03}.pt')
        end_time = time.time()

        logger([
            epoch,
            train_regularization, train_accuracy, train_loss, best['train_epoch'],
            test_regularization, test_accuracy, test_loss, best['test_epoch'],
            format_time(end_time - start_time),
            format_time((end_time - start_time) * (arguments.epochs - epoch) / epoch),
            format_time(end_time - epoch_time),
            *(scheduler.get_last_lr() if scheduler != None else []),
        ], save_path)

if __name__ == '__main__': main()
