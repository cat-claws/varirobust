import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms

from .datasets import MNIST

import io
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
    ])

train_set = MNIST('Dataset', train=True, download=True, transform=transform)
val_set = MNIST('Dataset', train=False, transform=transform)


def plot_trend(mat, title, xticks, yticks, xlabel, ylabel):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.0)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
    plt.yticks(np.arange(len(yticks)), yticks)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(mat.astype('float'), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = mat.max() / 2.
    for i, j in itertools.product(range(len(xticks)), range(len(yticks))):
        color = "white" if mat[i, j] > threshold else "black"
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

    decoded = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), -1)
    image = torch.from_numpy(decoded).permute(2, 0, 1)
    image = torch.cat((image[:3].flip(0), image[3:]))
    return image