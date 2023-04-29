import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms

import io
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from transformers.utils import ModelOutput

from . import datasets

transforms_3 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transforms_1 = transforms.ToTensor()

def auto_sets(name):
    if name == 'MNIST':
        train_set = datasets.auto_set('MNIST', download=True, train = True, transform=transforms_1)
        val_set = datasets.auto_set('MNIST', download=False, train = False, transform=transforms_1)
        channel = 1
    elif name == 'CIFAR10':
        train_set = datasets.auto_set('CIFAR10', download=True, train = True, transform=transforms_3)
        val_set = datasets.auto_set('CIFAR10', download=False, train = False, transform=transforms_1)
        channel = 3
    elif name == 'CIFAR100':
        train_set = datasets.auto_set('CIFAR100', download=True, train = True, transform=transforms_3)
        val_set = datasets.auto_set('CIFAR100', download=False, train = False, transform=transforms_1)
        channel = 3
    elif name == 'SVHN':
        train_set =  datasets.auto_set('SVHN', download=True, split = 'train', transform=transforms_1)
        val_set =  datasets.auto_set('SVHN', download=True, split = 'test', transform=transforms_1)
        channel = 3
    return train_set, val_set, channel


def xavier_init(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_normal_(net.weight, gain=nn.init.calculate_gain('relu'))
        if net.bias is not None:
            nn.init.zeros_(net.bias)
    elif isinstance(net, nn.Linear):
        nn.init.xavier_normal_(net.weight)
        if net.bias is not None:
            nn.init.zeros_(net.bias)
    # elif isinstance(net, nn.BatchNorm2d):
    #     net.weight.data.fill_(1)
    #     net.bias.data.zero_()

def certified_accuracy(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 100,
                    device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    cert = 0.
    cert_acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            print(counter)
            x_curr = x[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                        batch_size].to(device)

            output = model(x_curr)
            acc += (output.logits.max(1)[1] == y_curr).float().sum()
            cert += output.certified.float().sum()
            cert_acc += torch.logical_and(output.logits.max(1)[1] == y_curr, output.certified).float().sum()

    return ModelOutput(
        accuracy = acc.item() / x.shape[0],
        certified_rate = cert.item() / x.shape[0],
        certified_accuracy = cert_acc.item() / x.shape[0],
    )

from torchattacks import *
def attack_bundle(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 100,
                    device: torch.device = None):

    if device is None:
        device = x.device
    
    atks = [GN(model),
                FGSM(model, eps=8/255),
                AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False),
                Square(model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, verbose=False),
                BIM(model, eps=8/255, alpha=2/255, steps=10),
                RFGSM(model, eps=8/255, alpha=2/255, steps=10),
                PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True),
                EOTPGD(model, eps=8/255, alpha=2/255, steps=10, eot_iter=2),
                FFGSM(model, eps=8/255, alpha=10/255),
                TPGD(model, eps=8/255, alpha=2/255, steps=10),
                MIFGSM(model, eps=8/255, steps=10, decay=1.0),
                UPGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False),
                APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False),
                APGDT(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10),
                DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False),
                TIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False),
                Jitter(model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True),
                NIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0),
                PGDRS(model, eps=8/255, alpha=2/255, steps=10, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048),
                SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5),
                VMIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2),
                VNIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2),
                CW(model, c=1, kappa=0, steps=50, lr=0.01),
                PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True),
                PGDRSL2(model, eps=1.0, alpha=0.2, steps=10, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048),
                OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128),
                Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50),
                FAB(model, norm='Linf', steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9),
    ]
    adv_acc = {}
    for atk in atks:
        adv_acc[atk.attack] = 0.
        n_batches = math.ceil(x.shape[0] / batch_size)
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                        batch_size].to(device)

            adv_curr = atk(x_curr, y_curr)
            with torch.no_grad():

                output = model(adv_curr)
                adv_acc[atk.attack] += (output.max(1)[1] == y_curr).float().sum()

        adv_acc[atk.attack] = adv_acc[atk.attack].item() / x.shape[0]
        print(atk.attack, adv_acc[atk.attack])

    return ModelOutput(
        **adv_acc
    )





def kaiming_init(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0)
        elif len(param.shape) < 2:
            param.data.fill_(1)
        else:
            nn.init.kaiming_normal_(param)

def plot_trend(mat, title, xticks, yticks, xlabel, ylabel):
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(mat.T, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.0)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    plt.xticks(np.arange(len(xticks)), xticks)#, rotation=45)
    plt.yticks(np.arange(len(yticks)), np.around(yticks, decimals=2))

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(mat.astype('float'), decimals=4)

    # Use white text if squares are dark; otherwise black.
    threshold = mat.max() / 2.
    for i, j in itertools.product(range(len(xticks)), range(len(yticks))):
        color = "white" if mat[i, j] > threshold else "black"
        plt.text(i, j, labels[i, j], horizontalalignment="center", color=color)

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
