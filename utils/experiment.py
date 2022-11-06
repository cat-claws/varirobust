import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from .datasets import MNIST

from datetime import datetime
import io
import itertools
# from packaging import version


import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
    ])

train_set = MNIST('Dataset', train=True, download=True, transform=transform)
val_set = MNIST('Dataset', train=False, transform=transform)


def plot_trend(mat, title, xticks, yticks, xlabel, ylabel):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    plt.xticks(xticks, rotation=45)
    plt.yticks(yticks)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(mat.astype('float') / yticks.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = yticks.max() / 2.
    for i, j in itertools.product(range(yticks.shape[0]), range(yticks.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image