import argparse
import pathlib
import time
import torch

from data import cifar10
from learners import test, score
from resnet import resnet20
from utils import format_time

class Stiffness():
    def __init__(self, lagrange=1.):
        self.lagrange = lagrange

    def get_blocks(self, model, inputs):
        blocks_1, blocks_2, blocks_3, block_inputs_1, block_inputs_2, block_inputs_3 = [], [], [], [], [], []

        with torch.no_grad():
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
    def __init__(self, lagrange=1.): super(type(self), self).__init__(lagrange)

    def __call__(self, model, inputs):
        _b_1, _b_2, _b_3, b_i_1, b_i_2, b_i_3 = self.get_blocks(model, inputs)

        deltas = []
        for i in range(len(b_i_3)-1, 0, -1): deltas += [torch.norm(b_i_3[i] - b_i_3[i-1]) / torch.norm(b_i_3[i-1])]
        for i in range(len(b_i_2)-1, 0, -1): deltas += [torch.norm(b_i_2[i] - b_i_2[i-1]) / torch.norm(b_i_2[i-1])]
        for i in range(len(b_i_1)-1, 0, -1): deltas += [torch.norm(b_i_1[i] - b_i_1[i-1]) / torch.norm(b_i_1[i-1])]

        return torch.mean(torch.tensor(deltas)) * self.lagrange, None

class StiffnessIndex(Stiffness):
    def __init__(self, full=True, lagrange=1., validate=False, vectorize=True):
        super(type(self), self).__init__(lagrange)

        self.full = full
        self.validate = validate
        self.vectorize = vectorize

    def __call__(self, model, inputs):
        b_1, b_2, b_3, b_i_1, b_i_2, b_i_3 = self.get_blocks(model, inputs)
        blocks, block_inputs = b_1 + b_2 + b_3, b_i_1 + b_i_2 + b_i_3

        with torch.no_grad():
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

            return torch.mean(torch.tensor(stiffness_indices)) * self.lagrange, (outputs if self.validate else None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--checkpoint', default=150, type=int)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--inspect', default=20, type=int)
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction, default=False, type=bool)
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
    train_data, test_data = cifar10(arguments.batch_size, shuffle=False)

    test_accuracy, test_loss = test(model, test_data, device)
    print(test_accuracy, test_loss)

    stiffness_index = StiffnessIndex(validate=arguments.validate)
    stiffness_index = TotalNeuralStiffness()

    for i, (inputs, labels) in enumerate(train_data):
        start_time = time.time()
        full_stiffness_index, outputs = stiffness_index(model, inputs.to(device))
        print('full_stiffness_index', full_stiffness_index.item(), format_time(time.time() - start_time), end=' ')
        if arguments.validate: print(score(outputs, labels.to(device)), labels.item())
        else: print()
        if i == arguments.inspect-1: break

if __name__ == '__main__': main()
