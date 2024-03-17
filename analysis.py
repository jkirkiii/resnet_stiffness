import argparse
import functorch
import pathlib
import time
import torch

from data import cifar10
from resnet import resnet20
from learners import test
from utils import format_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--checkpoint', default=150, type=int)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--model', default='resnet20', type=str)
    arguments = parser.parse_args()

    model, model_path = None, None

    if arguments.model == 'resnet20': model, model_path = resnet20(), 'resnet20'
    else: pass

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if arguments.device == '' else arguments.device
    model.to(device)
    model.eval()

    load_path = pathlib.Path('.') / 'experiments' / model_path / 'checkpoints' / f'{arguments.checkpoint:03}.pt'
    model.load_state_dict(torch.load(load_path, map_location=torch.device(device)), strict=False)
    print('load', load_path)
    train_data, test_data = cifar10(arguments.batch_size)

    # test_accuracy, test_los_s_ = test(model, test_data, device)
    # print(test_accuracy, test_loss)

    train_iter = iter(train_data)
    inputs, labels = next(train_iter)
    print('inputs', list(inputs.shape), 'labels', list(labels.shape))

    with torch.no_grad():
        out = torch.nn.functional.relu(model.bn1(model.conv1(inputs.to(device))))
        out = model.layer1[0](out)
        out = model.layer1[1](out)
        out = model.layer1[2](out)
        out = model.layer2[0](out)
        out = model.layer2[1](out)
        out = model.layer2[2](out)
        out = model.layer3[0](out)
        out = model.layer3[1](out)
        block_inputs = out
        block = model.layer3[2]
        print('block')
        print(block)
        print('block_inputs', list(block_inputs.shape))


        _s_ = time.time()
        jacobian = torch.autograd.functional.jacobian(block, block_inputs)
        _e_ = time.time()
        print('torch.autograd.functional.jacobian', list(jacobian.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/functorch/stable/generated/functorch.jacfwd.html
        #   note: functorch is deprecated
        _s_ = time.time()
        jacobian = torch.func.jacfwd(block)(block_inputs)
        _e_ = time.time()
        print('torch.func.jacfwd', list(jacobian.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/docs/stable/generated/torch.func.jacrev.html
        _s_ = time.time()
        jacobian = torch.func.jacrev(block)(block_inputs)
        _e_ = time.time()
        print('torch.func.jacrev', list(jacobian.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/docs/stable/generated/torch.linalg.eig.html
        _s_ = time.time()
        eigenvalues, eigenvectors = torch.linalg.eig(jacobian)
        _e_ = time.time()
        print('torch.linalg.eig', 'eigenvalues', list(eigenvalues.shape), 'eigenvectors', list(eigenvectors.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/docs/stable/generated/torch.linalg.eigvals.html
        _s_ = time.time()
        eigenvalues = torch.linalg.eigvals(jacobian)
        _e_ = time.time()
        print('torch.linalg.eigvals', list(eigenvalues.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html
        # _s_ = time.time()
        # eigenvalues, eigenvectors = torch.linalg.eigh(jacobian)
        # _e_ = time.time()
        # print('torch.linalg.eigh', 'eigenvalues', list(eigenvalues.shape), 'eigenvectors', list(eigenvectors.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
        _s_ = time.time()
        u, s, v = torch.linalg.svd(jacobian)
        _e_ = time.time()
        print('torch.linalg.svd', 'u', list(u.shape), 's', list(s.shape), 'v', list(v.shape), format_time(_e_ - _s_, fine=True))


        #   https://pytorch.org/docs/stable/generated/torch.linalg.qr.html
        _s_ = time.time()
        q, r = torch.linalg.qr(jacobian)
        _e_ = time.time()
        print('torch.linalg.qr', 'q', list(q.shape), 'r', list(r.shape), format_time(_e_ - _s_, fine=True))

    #   https://anonymous.4open.science/r/Stiffness-Analysis-ResNets-18D2
    return

if __name__ == '__main__': main()
