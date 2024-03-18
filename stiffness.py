import argparse
import functorch
import pathlib
import time
import torch

from data import cifar10
from resnet import resnet20
from learners import test, score
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
    train_data, test_data = cifar10(arguments.batch_size, shuffle=False)

    test_accuracy, test_loss = test(model, test_data, device)
    print(test_accuracy, test_loss)

    # train_iter = iter(train_data)
    # inputs, labels = next(train_iter)
    # print('inputs', list(inputs.shape), 'labels', list(labels.shape))

    for inputs, labels in train_data:
        _0 = time.time()

        with torch.no_grad():
            blocks, block_inputs = [], []

            out = torch.nn.functional.relu(model.bn1(model.conv1(inputs.to(device))))
            block_inputs += [out]

            for i, block in enumerate(model.layer1):
                out = block(out)
                blocks += [block.eval()]
                block_inputs += [out.clone().detach()]

            for i, block in enumerate(model.layer2):
                out = block(out)
                blocks += [block.eval()]
                block_inputs += [out.clone().detach()]

            for i, block in enumerate(model.layer3):
                out = block(out)
                blocks += [block.eval()]
                block_inputs += [out.clone().detach()]

            stiffness_indices = []

            for i, (block, block_input) in enumerate(zip(blocks, block_inputs[:-1])):
                jacobian = torch.func.jacrev(block)(block_input)
                eigenvalues = torch.linalg.eigvals(jacobian)
                eigen_max = torch.max(torch.real(eigenvalues))
                eigen_min = torch.min(torch.real(eigenvalues))
                block_stiffness_index = eigen_max / eigen_min
                stiffness_indices += [block_stiffness_index]

                # print(
                #     'block_stiffness_index', block_stiffness_index.item(),
                #     'eigen_max', eigen_max.item(),
                #     'eigen_min', eigen_min.item(),
                # )

            _out = block_inputs[9]
            _out = torch.nn.functional.avg_pool2d(_out, _out.size()[3])
            _out = _out.view(_out.size(0), -1)
            _out = model.linear(_out)

            print(
                'full_stiffness_index',
                torch.mean(torch.tensor(stiffness_indices)).item(),
                format_time(time.time() - _0),
                score(_out, labels),
            )

        # break

if __name__ == '__main__': main()
