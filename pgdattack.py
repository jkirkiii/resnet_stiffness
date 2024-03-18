import random
import torch
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from resnet import resnet20
from data import cifar10
from utils import format_time


# to test code in this file
def main():
    # load data
    train_data, test_data = cifar10(256)

    # load model checkpoint
    model = resnet20()
    checkpoints_path = pathlib.Path('.') / 'experiments' / 'resnet20' / 'checkpoints' / '150.pt'
    state_dict = torch.load(checkpoints_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # create adversarial images and save them to files
    root_path = pathlib.Path('.') / 'experiments'
    for i, (inputs, labels) in enumerate(train_data):
        _s_ = time.time()
        adv_images = pgd_attack(model, torch.nn.CrossEntropyLoss(), inputs, labels, 1, .1, 10)
        adv_images_path = root_path / f'adv_images{i + 1:03d}'
        save_adv_images(adv_images, adv_images_path)
        _e_ = time.time()
        print(format_time(_e_ - _s_, fine=True))
        show_images(root_path / (adv_images_path.name + '.npy'))

    # filenames = [str(p) for p in root_path.glob('adv_images*.npy')]
    # dataset = AdversarialImageDataset(filenames)


def pgd_attack(model, objective, images, labels, epsilon, alpha, num_iters):
    """
    Perform Projected Gradient Descent (PGD) attack on the given model.

    Args:
        model (torch.nn.Module): The model to be attacked.
        objective (function(outputs, labels)): Loss function to be
        images (torch.Tensor): Input images for the attack.
        labels (torch.Tensor): True labels for the images.
        epsilon (float): Perturbation size for each step.
        alpha (float): Step size of the attack.
        num_iters (int): Number of iterations for the attack.

    Returns:
        adv_images (torch.Tensor): Adversarial examples generated using PGD.
    """
    adv_images = images.clone().detach().requires_grad_(True)

    for i in range(num_iters):
        outputs = model(adv_images)
        loss = objective(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images_grad = adv_images.grad.data

        # PGD step
        adv_images.data = adv_images.data + alpha * adv_images_grad.sign()

        # Projecting the perturbation to the epsilon ball around the original image
        eta = torch.clamp(adv_images.data - images.data, -epsilon, epsilon)
        adv_images.data = images.data + eta
        adv_images.data = torch.clamp(adv_images.data, 0, 1)  # Clip to valid pixel range [0, 1]

    return adv_images


def save_adv_images(adv_images, filename):
    np_images = adv_images.detach().cpu().numpy()
    np.save(filename, np_images)

    print(f"Adversarial images saved to: {filename}")


def show_images(filename, num_images=10, num_cols=5, figsize=(10, 10)):
    images = np.load(filename)
    random_indices = random.sample(range(len(images)), num_images)
    selected_images = images[random_indices]
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(f"Images from {filename}", fontsize=12)

    for i in range(num_images):
        row = int(i / num_cols)
        col = i % num_cols
        image = selected_images[i].transpose(1, 2, 0)  # Reshape to (height, width, channels)
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

    # # Hide extra axes if there are fewer images than the grid can hold
    # for r in range(num_rows):
    #     for c in range(num_cols):
    #         if r * num_cols + c >= num_images:
    #             axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()


class AdversarialImageDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        # Assuming all files contain the same number of images
        with open(self.filenames[0], 'rb') as f:
            data = np.load(f)
        return data.shape[0]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        images = np.load(filename)
        images = np.expand_dims(images, axis=0)  # Add sample dimension
        image_tensor = torch.from_numpy(images).float()

        return image_tensor


if __name__ == '__main__': main()
