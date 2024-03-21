import argparse
import numpy as np
import pathlib
import time
import torch

from data import cifar10
from learners import test, score
from resnet import resnet20
from utils import format_time

def get_jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 

    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.                                                                                

    return torch.stack(jac).reshape(y.shape + x.shape)

class Stiffness():
    def __init__(self, lagrange=1.):
        self.lagrange = lagrange

    def get_blocks(self, model, inputs):
        blocks_1, blocks_2, blocks_3, block_inputs_1, block_inputs_2, block_inputs_3 = [], [], [], [], [], []

        # with torch.no_grad():
        outputs = torch.nn.functional.relu(model.bn1(model.conv1(inputs)))
        block_inputs_1 += [outputs]

        for block in model.layer1:
            outputs = block(outputs)
            blocks_1 += [block]
            block_inputs_1 += [outputs]

        for block in model.layer2:
            outputs = block(outputs)
            blocks_2 += [block]
            block_inputs_2 += [outputs]

        for block in model.layer3:
            outputs = block(outputs)
            blocks_3 += [block]
            block_inputs_3 += [outputs]

        return blocks_1, blocks_2, blocks_3, block_inputs_1, block_inputs_2, block_inputs_3

class TotalNeuralStiffness(Stiffness):
    def __init__(self, lagrange=1., per_block=True):
        super(type(self), self).__init__(lagrange)
        self.per_block = per_block

    def __call__(self, model, inputs):
        _b_1, _b_2, _b_3, b_i_1, b_i_2, b_i_3 = self.get_blocks(model, inputs)

        deltas = []
        for i in range(len(b_i_3)-1, 0, -1): deltas += [torch.linalg.vector_norm(b_i_3[i] - b_i_3[i-1]) / torch.linalg.vector_norm(b_i_3[i-1])]
        for i in range(len(b_i_2)-1, 0, -1): deltas += [torch.linalg.vector_norm(b_i_2[i] - b_i_2[i-1]) / torch.linalg.vector_norm(b_i_2[i-1])]
        for i in range(len(b_i_1)-1, 0, -1): deltas += [torch.linalg.vector_norm(b_i_1[i] - b_i_1[i-1]) / torch.linalg.vector_norm(b_i_1[i-1])]

        return torch.mean(torch.tensor(deltas)) * self.lagrange, torch.tensor(deltas) if self.per_block else None

class StiffnessIndex(Stiffness):
    def __init__(self, full=True, lagrange=1., validate=False, vectorize=True):
        super(type(self), self).__init__(lagrange)

        self.full = full
        self.validate = validate
        self.vectorize = vectorize

    def __call__(self, model, inputs):
        b_1, b_2, b_3, b_i_1, b_i_2, b_i_3 = self.get_blocks(model, inputs)
        blocks, block_inputs = b_1 + b_2 + b_3, b_i_1 + b_i_2 + b_i_3

        # with torch.no_grad():
        stiffness_indices = []

        for block, block_input in zip(blocks, block_inputs[:-1]):            
            jacobian = torch.func.jacrev(block)(block_input) if self.vectorize else torch.autograd.functional.jacobian(block, block_input)
            eigen_values = torch.linalg.eigvals(jacobian)
            eigen_max = torch.max(torch.real(eigen_values))
            eigen_min = torch.min(torch.real(eigen_values))
            block_stiffness_index = torch.abs(eigen_max / eigen_min) if self.full else torch.abs(eigen_max)
            stiffness_indices += [block_stiffness_index]

        if self.validate:
            outputs = block_inputs[-1]
            outputs = torch.nn.functional.avg_pool2d(outputs, outputs.size()[3])
            outputs = outputs.view(outputs.size(0), -1)
            outputs = model.linear(outputs)

        return torch.mean(torch.tensor(stiffness_indices)) * self.lagrange, outputs if self.validate else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('-c', '--checkpoint', default=150, type=int)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--division', default=1, type=int)
    parser.add_argument('-e', '--evaluate', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('-f', '--full', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('-i', '--inspect', default=1, type=int)
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('-n', '--name', default='', type=str)
    parser.add_argument('--num-divisions', default=1, type=int)
    parser.add_argument('-p', '--power-iterations', default=1, type=int)
    parser.add_argument('--spectral-norm', action=argparse.BooleanOptionalAction, default=False, type=bool)
    parser.add_argument('-s', '--stiffness-index', default='true', type=str)
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction, default=False, type=bool)
    arguments = parser.parse_args()

    model = None

    if arguments.model == 'resnet20': model = resnet20(power_iterations=arguments.power_iterations, spectral_norm=arguments.spectral_norm)
    else: pass

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if arguments.device == '' else arguments.device
    model.to(device)
    model.eval()

    load_path = pathlib.Path('.') / 'experiments' / arguments.name / 'checkpoints' / f'{arguments.checkpoint:03}.pt'
    model.load_state_dict(torch.load(load_path, map_location=torch.device(device)), strict=False)
    print('load', load_path)
    train_data, test_data = cifar10(arguments.batch_size, shuffle=True)

    if arguments.evaluate:
        test_accuracy, test_loss, _, _ = test(model, test_data, device)
        print(test_accuracy, test_loss)

    if arguments.stiffness_index == 'true': stiffness_index = StiffnessIndex(full=arguments.full, validate=arguments.validate)
    elif arguments.stiffness_index == 'total': stiffness_index = TotalNeuralStiffness()
    else: pass

    stiffnesses = []

    for i, (inputs, labels) in enumerate(train_data):
        start_time = time.time()
        full_stiffness_index, outputs = stiffness_index(model, inputs.to(device))
        stiffnesses += [full_stiffness_index.item()]
        print(arguments.stiffness_index, full_stiffness_index.item(), format_time(time.time() - start_time), end=' ')
        if arguments.validate: print(score(outputs, labels.to(device)), labels.item())
        else: print()
        if i == arguments.inspect-1: break

    if arguments.inspect > 1:
        stiffnesses = np.array(stiffnesses)
        stiffnesses_mean = np.mean(stiffnesses)
        stiffnesses_std = np.std(stiffnesses)
        print(stiffnesses_mean, stiffnesses_std)

if __name__ == '__main__': main()
