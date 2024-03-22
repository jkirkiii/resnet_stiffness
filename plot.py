import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from utils import generate_color_transition, generate_random_colors

def get_min(losses):
    min_losses, min_loss = [], float('inf')

    for i in range(len(losses)):
        if losses[i] < min_loss: min_loss = losses[i]
        min_losses += [min_loss]

    return np.array(min_losses)

def plot_training(
        path: str, *,
        accuracy_colors=[],
        checkpoints=[],
        labels=[],
        length=1,
        loss_colors=[],
        scale=2,
        show=['test', 'train'],
):
    figure, loss_plt = plt.subplots(facecolor='white', figsize=(4*scale, 3*scale))
    accuracy_plt = loss_plt.twinx()
    regularization_plt = loss_plt.twinx()

    experiment_path = pathlib.Path(path)

    random_colors = generate_random_colors(length)

    for i, (label, loss_color, accuracy_color) in enumerate(zip(
        labels if labels != [] else range(1, length+1),
        loss_colors if loss_colors != [] else random_colors,
        accuracy_colors if accuracy_colors != [] else random_colors,
    )):
        log_path = experiment_path / 'log.csv'

        rows = None

        with open(log_path, 'r') as file:
            reader = csv.reader(file)
            rows = np.array([row for row in reader])

        print('load', log_path, rows.shape)

        epochs = np.array(rows[:, 0], dtype=int)
        train_accuracy = np.array(rows[:, 1], dtype=float)
        train_loss = np.array(rows[:, 2], dtype=float)
        # best_train_epoch = np.array(rows[:, 3], dtype=int)
        test_accuracy = np.array(rows[:, 4], dtype=float)
        test_loss = np.array(rows[:, 5], dtype=float)
        # best_test_epoch = np.array(rows[:, 6], dtype=int)
        # time_elapsed = np.array(rows[:, 7], dtype=str)
        # time_left = np.array(rows[:, 8], dtype=str)
        # epoch_time = np.array(rows[:, 9], dtype=str)
        # learning_rate = np.array(rows[:, 10], dtype=float)
        # train_regularization = np.array(rows[:, 11], dtype=float)

        deltas = []
        deltas += [np.array(rows[:, 18], dtype=float)]
        deltas += [np.array(rows[:, 17], dtype=float)]
        deltas += [np.array(rows[:, 16], dtype=float)]
        deltas += [np.array(rows[:, 15], dtype=float)]
        deltas += [np.array(rows[:, 14], dtype=float)]
        deltas += [np.array(rows[:, 13], dtype=float)]
        deltas += [np.array(rows[:, 12], dtype=float)]

        if 'regularization' in show:
            for i, (delta, color) in enumerate(zip(
                deltas,
                generate_random_colors(len(deltas)),
                # generate_color_transition('#00ff00', '#ff00ff', len(deltas)),
            )):
                regularization_plt.plot(epochs, delta,  alpha=.8, color=color, label=f'{i+1}', linestyle='solid')

            regularization_plt.axis('off')

        if 'test' in show:
            loss_plt.plot(epochs, test_loss, alpha=.8, color=loss_color, linestyle='solid', label='test loss')
            loss_plt.plot(epochs, get_min(test_loss), alpha=.4, color=loss_color, linestyle='dotted')
            if 'accuracy' in show: accuracy_plt.plot(epochs, test_accuracy, alpha=.8, color=accuracy_color, linestyle='solid', label='test acc.')

        if 'train' in show:
            loss_plt.plot(epochs, train_loss, alpha=.4, color=loss_color, linestyle='dashed', label='train loss')
            if 'accuracy' in show: accuracy_plt.plot(epochs, train_accuracy, alpha=.4, color=accuracy_color, linestyle='dashed', label='train acc.')

    if 'checkpoints' in show:
        for checkpoint in checkpoints: plt.axvline(x=checkpoint, alpha=.8, color='k', label=str(checkpoint), linestyle='--')

    loss_plt.set_xlabel('Epochs')
    loss_plt.set_ylabel('Loss')
    accuracy_plt.set_ylabel('Accuracy')
    regularization_plt.legend()
    figure.tight_layout()
    figure.savefig(experiment_path / 'training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='', type=str)
    arguments = parser.parse_args()

    plot_training(
        f'experiments/{arguments.name}',
        show=['regularization'],
    )
