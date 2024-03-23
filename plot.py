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
    figure, regularization_plt = plt.subplots(facecolor='white', figsize=(4*scale, 3*scale))
    accuracy_plt = regularization_plt.twinx()
    loss_plt = regularization_plt.twinx()

    experiment_path = pathlib.Path(path)

    random_colors = generate_random_colors(length)

    for _i, (label, loss_color, accuracy_color) in enumerate(zip(
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
        # train_deltas = [np.array(rows[:, 11+j+1], dtype=float) for j in range(7)]
        train_deltas = []
        # train_deltas += [np.array(rows[:, 11+1], dtype=float)]
        train_deltas += [np.array(rows[:, 11+2], dtype=float)]
        train_deltas += [np.array(rows[:, 11+3], dtype=float)]
        train_deltas += [np.array(rows[:, 11+4], dtype=float)]
        train_deltas += [np.array(rows[:, 11+5], dtype=float)]
        train_deltas += [np.array(rows[:, 11+6], dtype=float)]
        train_deltas += [np.array(rows[:, 11+7], dtype=float)]
        train_deltas = list(reversed(train_deltas))
        # test_regularization = np.array(rows[:, 19], dtype=float)
        # test_deltas = [np.array(rows[:, 19+j+1], dtype=float) for j in range(7)]

        if 'regularization' in show:
            for i, (delta, color) in enumerate(zip(
                train_deltas,
                # generate_random_colors(7),
                generate_color_transition('#0000FF', '#FF0000', 7),
            )):
                regularization_plt.plot(epochs, delta,  alpha=.9, color=color, label=f'{i+1}', linestyle='solid')

            # regularization_plt.axis('off')

        if 'test' in show:
            loss_plt.plot(epochs, test_loss, alpha=.4, color=loss_color, linestyle='dashed', label='test loss')
            # loss_plt.plot(epochs, get_min(test_loss), alpha=.3, color=loss_color, linestyle='dotted')
            if 'accuracy' in show: accuracy_plt.plot(epochs, test_accuracy, alpha=.4, color=accuracy_color, linestyle='dashed', label='test acc.')

        if 'train' in show:
            loss_plt.plot(epochs, train_loss, alpha=.4, color=loss_color, linestyle='dotted', label='train loss')
            if 'accuracy' in show: accuracy_plt.plot(epochs, train_accuracy, alpha=.4, color=accuracy_color, linestyle='dotted', label='train acc.')

    if 'checkpoints' in show:
        for checkpoint in checkpoints: plt.axvline(x=checkpoint, alpha=.4, color='k', label=str(checkpoint), linestyle='--')

    regularization_plt.set_xlabel('Epochs')
    loss_plt.axis('off')
    accuracy_plt.axis('off')
    regularization_plt.set_ylabel('Total Neural Stiffness')
    regularization_plt.set_ylim([.00075, .0045]) # .011
    regularization_plt.legend()
    figure.tight_layout()
    figure.savefig(experiment_path / 'training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='', type=str)
    arguments = parser.parse_args()

    plot_training(
        f'experiments/{arguments.name}',
        show=[
            'accuracy',
            'regularization',
            'test',
            'train',
        ],
        loss_colors=['#ff00ff'],
        accuracy_colors=['#00ff00'],
    )
