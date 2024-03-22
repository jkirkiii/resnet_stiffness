import argparse
import datetime
import pathlib
import time
import torch
import torch.cuda

from resnet import resnet20

from data import cifar10
from learners import test, train
from utils import format_time, logger
from stiffness import StiffnessIndex, TotalNeuralStiffness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--division', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('-l', '--lagrange', default=0., type=float)
    parser.add_argument('--learning-rate', default=.1, type=float, help='initial learning rate')
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('--momentum', default=.9, type=float, help='momentum (default: .9)')
    parser.add_argument('-n', '--name', default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    parser.add_argument('--num-divisions', default=1, type=int)
    parser.add_argument('-p', '--power-iterations', default=30, type=int)
    parser.add_argument('--pgd', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('--spectral-norm', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('-s', '--sample', default=1, type=int)
    parser.add_argument('-w', '--weight-decay', default=5.e-4, type=float, help='weight decay (default: 5.e-4)')
    arguments = parser.parse_args()

    model = None

    if arguments.model == 'resnet20': model = resnet20(power_iterations=arguments.power_iterations, spectral_norm=arguments.spectral_norm)
    else: pass

    train_data, test_data = cifar10(arguments.batch_size, arguments.num_divisions, arguments.division)
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=arguments.learning_rate, momentum=arguments.momentum, weight_decay=arguments.weight_decay)
    regularizer = StiffnessIndex(lagrange=arguments.lagrange) if arguments.lagrange > 0. else None
    index = TotalNeuralStiffness()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=10)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if arguments.device == '' else arguments.device
    model.to(device)

    save_path = pathlib.Path('.') / 'experiments' / arguments.name
    checkpoints_path = save_path / 'checkpoints'
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path.mkdir(exist_ok=True)
    print('make', save_path)

    best = {'train_loss': float('inf'), 'train_epoch': 0, 'test_loss': float('inf'), 'test_epoch': 0}
    start_time = time.time()

    for epoch in range(1, arguments.epochs+1):
        epoch_time = time.time()
        train_accuracy, train_loss, train_regularization, train_deltas = train(model, train_data, device, objective, optimizer, index, regularizer, arguments.sample, arguments.pgd)
        test_accuracy, test_loss, test_regularization, test_deltas = test(model, test_data, device, objective, index, regularizer, arguments.sample)
        scheduler.step(train_loss)
        if train_loss < best['train_loss']: best['train_loss'], best['train_epoch'] = train_loss, epoch
        if test_loss < best['test_loss']: best['test_loss'], best['test_epoch']= test_loss, epoch
        torch.save(model.state_dict(), checkpoints_path / f'{epoch:03}.pt')
        end_time = time.time()

        logger([
            epoch,
            train_accuracy, train_loss, best['train_epoch'], 
            test_accuracy, test_loss, best['test_epoch'],
            format_time(end_time - start_time),
            format_time((end_time - start_time) * (arguments.epochs - epoch) / epoch),
            format_time(end_time - epoch_time),
            *(scheduler.get_last_lr() if scheduler != None else []),
        ], save_path, silent=[
            train_regularization, *(train_deltas.tolist() if train_deltas != None else []),
            test_regularization, *(test_deltas.tolist() if test_deltas != None else []),
        ])

if __name__ == '__main__': main()
