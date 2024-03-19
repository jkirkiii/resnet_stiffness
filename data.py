import torch
import torchvision
import numpy as np


def cifar10(batch_size, shuffle=True, num_divisions=1, division=1):
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
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    if num_divisions > 1:
        divisions = np.array_split(np.arange(len(train_dataset)), num_divisions)
        indices = np.array(divisions[division - 1])
        subset = torch.utils.data.Subset(train_dataset, indices=indices)
        train_data = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.Compose(train_transforms), download=True)
    # test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.Compose(test_transforms), download=True)
    # train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # test_data = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_data, test_data
