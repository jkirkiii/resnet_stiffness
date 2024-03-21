'''
Properly implemented ResNet-s for CIFAR-10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2] which
is implemented for ImageNet and doesn't have option A for identity. Moreover,
most of the implementations on the web is copy-paste from torchvision's ResNet
and has wrong number of parameters.

Proper ResNet-s for CIFAR-10 (for fair comparision etc.) has the following
number of layers and parameters:

name       | layers | params
ResNet20   |     20 |   .27M
ResNet32   |     32 |   .46M
ResNet44   |     44 |   .66M
ResNet56   |     56 |   .85M
ResNet110  |    110 |  1.7M
ResNet1202 |   1202 | 19.4M

which this implementation indeed has.

References:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch


def initialize_weight(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d): torch.nn.init.kaiming_normal_(
        layer.weight)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambda_):
        super(type(self), self).__init__()
        self.lambda_ = lambda_

    def forward(self, x): return self.lambda_(x)


class SpectralNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, *, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        torch.nn.utils.parametrizations.spectral_norm(self.conv, n_power_iterations=power_iterations)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', *, power_iterations=1, use_spectral_norm=False):
        super(BasicBlock, self).__init__()

        if use_spectral_norm:
            self.conv1 = SpectralNorm(in_planes, planes, kernel_size=3, stride=stride, padding=1, power_iterations=power_iterations)
            self.conv2 = SpectralNorm(planes, planes, kernel_size=3, stride=1, padding=1, power_iterations=power_iterations)
        else:
            self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer(
                    lambda x: torch.nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                      'constant', 0))
            elif option == 'B':
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, *, power_iterations=1, use_spectral_norm=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.use_spectral_norm = use_spectral_norm
        self.power_iterations = power_iterations

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = torch.nn.Linear(64, num_classes)

        self.apply(initialize_weight)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers, strides = [], [stride] + [1] * (num_blocks - 1)

        for stride in strides:
            layers += [block(self.in_planes, planes, stride, use_spectral_norm=self.use_spectral_norm)]
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(*, power_iterations=1, spectral_norm=False): return ResNet(BasicBlock, [3, 3, 3], power_iterations=power_iterations, use_spectral_norm=spectral_norm)


def resnet32(): return ResNet(BasicBlock, [5, 5, 5])


def resnet44(): return ResNet(BasicBlock, [7, 7, 7])


def resnet56(): return ResNet(BasicBlock, [9, 9, 9])


def resnet110(): return ResNet(BasicBlock, [18, 18, 18])


def resnet1202(): return ResNet(BasicBlock, [200, 200, 200])
