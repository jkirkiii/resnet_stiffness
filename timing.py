import argparse
import pathlib
import time
import torch

from data import cifar10
from resnet import resnet20
from learners import test
from utils import format_time
from stiffness import gelfand, get_jacobian, power_iteration

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--division', default=1, type=int)
    parser.add_argument('-e', '--evaluate', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('-n', '--name', default='', type=str)
    parser.add_argument('--num-divisions', default=1, type=int)
    parser.add_argument('-p', '--power-iterations', default=10, type=int)
    arguments = parser.parse_args()

    model = None

    if arguments.model == 'resnet20': model = resnet20()
    else: pass

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if arguments.device == '' else arguments.device
    model.to(device)
    model.eval()

    load_path = pathlib.Path('.') / 'experiments' / arguments.name / 'checkpoints' / f'{arguments.checkpoint:03}.pt'
    model.load_state_dict(torch.load(load_path, map_location=torch.device(device)), strict=False)
    print('load', load_path)
    train_data, test_data = cifar10(batch_size=arguments.batch_size, shuffle=True)

    if arguments.evaluate:
        test_accuracy, test_loss, _, _ = test(model, test_data, device)
        print(test_accuracy, test_loss)

    train_iter = iter(train_data)
    inputs, labels = next(train_iter)
    print('inputs', list(inputs.shape), 'labels', list(labels.shape))


    #   method from https://anonymous.4open.science/r/Stiffness-Analysis-ResNets-18D2
    out = torch.nn.functional.relu(model.bn1(model.conv1(inputs.to(device))))
    block = model.layer1[0]
    # out = model.layer1[0](out)
    # out = model.layer1[1](out)
    # out = model.layer1[2](out)
    # out = model.layer2[0](out)
    # out = model.layer2[1](out)
    # out = model.layer2[2](out)
    # out = model.layer3[0](out)
    # block = model.layer3[1]
    # out = model.layer3[1](out)
    block_inputs = out
    # print('block')
    # print(block)
    print('block_inputs', list(block_inputs.shape))


    t = time.time()
    jacobian = torch.func.jacrev(block)(block_inputs)
    t = time.time() - t
    print(format_time(t, fine=True), torch.linalg.vector_norm(jacobian).item(), 'torch.func.jacrev()', list(jacobian.shape), jacobian.numel())

    # t = time.time()
    # eigen_values, _eigen_vectors = torch.linalg.eig(jacobian)
    # eigen_max = torch.max(torch.real(eigen_values))
    # t = time.time() - t
    # print(format_time(t, fine=True), eigen_max.item(), 'torch.linalg.eig(torch.max(torch.real((()))')

    t = time.time()
    eigen_values = torch.linalg.eigvals(jacobian)
    eigen_max = torch.max(torch.real(eigen_values))
    t = time.time() - t
    print(format_time(t, fine=True), eigen_max.item(), 'torch.linalg.eigvals(torch.max(torch.real((()))')

    t = time.time()
    eigen_max, _ = power_iteration(jacobian, device)
    t = time.time() - t
    print(format_time(t, fine=True), eigen_max.item(), 'power_iteration()')

    t = time.time()
    eigen_max = gelfand(jacobian)
    t = time.time() - t
    print(format_time(t, fine=True), eigen_max.item(), 'gelfands()')

    # for i in range(0, arguments.power_iterations, 5):
    #     t = time.time()
    #     eigen_max, _ = power_iteration(jacobian, device, iterations=i+1)
    #     t = time.time() - t
    #     print(format_time(t, fine=True), eigen_max.item(), 'power_iteration()', i+1)

    # for i in range(arguments.power_iterations):
    #     t = time.time()
    #     eigen_max = gelfands(jacobian, iterations=i+1)
    #     t = time.time() - t
    #     print(format_time(t, fine=True), eigen_max.item(), 'gelfands', i+1)


    return


    t = time.time()
    jacobian = get_jacobian(block(block_inputs), block_inputs)
    t = time.time() - t
    print(format_time(t, fine=True), torch.linalg.vector_norm(jacobian).item(), 'get_jacobian', list(jacobian.shape))


    t = time.time()
    jacobian = torch.autograd.functional.jacobian(block, block_inputs)
    t = time.time() - t
    print(format_time(t, fine=True), torch.linalg.vector_norm(jacobian).item(), 'torch.autograd.functional.jacobian', list(jacobian.shape))


    #   https://pytorch.org/functorch/stable/generated/functorch.jacfwd.html
    #   note: functorch is deprecated
    t = time.time()
    jacobian = torch.func.jacfwd(block)(block_inputs)
    t = time.time() - t
    print(format_time(t, fine=True), torch.linalg.vector_norm(jacobian).item(), 'torch.func.jacfwd', list(jacobian.shape))


    #   https://pytorch.org/docs/stable/generated/torch.func.jacrev.html
    t = time.time()
    jacobian = torch.func.jacrev(block)(block_inputs)
    t = time.time() - t
    print(format_time(t, fine=True), torch.linalg.vector_norm(jacobian).item(), 'torch.func.jacrev', list(jacobian.shape))


    #   https://pytorch.org/docs/stable/generated/torch.linalg.eig.html
    t = time.time()
    eigenvalues, eigenvectors = torch.linalg.eig(jacobian)
    t = time.time() - t
    print(format_time(t, fine=True), 'torch.linalg.eig', 'eigenvalues', list(eigenvalues.shape), 'eigenvectors', list(eigenvectors.shape))


    #   https://pytorch.org/docs/stable/generated/torch.linalg.eigvals.html
    t = time.time()
    eigenvalues = torch.linalg.eigvals(jacobian)
    t = time.time() - t
    print(format_time(t, fine=True), 'torch.linalg.eigvals', list(eigenvalues.shape))


    # #   https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html
    # t = time.time()
    # eigenvalues, eigenvectors = torch.linalg.eigh(jacobian)
    # t = time.time() - t
    # print(format_time(t, fine=True), 'torch.linalg.eigh', 'eigenvalues', list(eigenvalues.shape), 'eigenvectors', list(eigenvectors.shape))


    #   https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
    t = time.time()
    u, s, v = torch.linalg.svd(jacobian)
    t = time.time() - t
    print(format_time(t, fine=True), 'torch.linalg.svd', 'u', list(u.shape), 's', list(s.shape), 'v', list(v.shape))


    #   https://pytorch.org/docs/stable/generated/torch.linalg.qr.html
    t = time.time()
    q, r = torch.linalg.qr(jacobian)
    t = time.time() - t
    print(format_time(t, fine=True), 'torch.linalg.qr', 'q', list(q.shape), 'r', list(r.shape))

if __name__ == '__main__': main()
