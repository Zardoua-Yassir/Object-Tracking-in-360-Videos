"""
Script to transform tensorboard files into matplotlib graphs
"""

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Replace with the actual path to your log directory
logdir = 'C:/Users/Dell/Desktop/AR Project 7-21-2024/SiamCAR-360 - Smooth/logs/Losses_Epoch_weighted'
output_dir = 'C:/Users/Dell/Desktop/AR Project 7-21-2024/SiamCAR-360 - Smooth/output_figures'
os.makedirs(output_dir, exist_ok=True)


def save_figure(event_acc, tag, output_path):
    """Save figure for a specific tag."""
    steps = [0]
    values = [7.20]
    for event in event_acc.Scalars(tag):
        steps.append(event.step + 1)
        values.append(event.value)

    plt.figure()
    # plt.plot(steps, values, label=tag)
    plt.plot(steps, values, label="Weighted Loss")
    # plt.xlabel('Steps')
    plt.xlabel('Epoch')
    # plt.ylabel('Value')
    plt.ylabel('Loss value')
    plt.title(tag)
    plt.legend()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close()


def main():
    # Initialize EventAccumulator
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Get all tags
    tags = event_acc.Tags()['scalars']

    # Save figures for each tag
    for tag in tags:
        output_path = os.path.join(output_dir, f'{tag.replace("/", "_")}.png')
        save_figure(event_acc, tag, output_path)
        print(f'Saved figure for tag {tag} to {output_path}')


if __name__ == '__main__':
    main()
