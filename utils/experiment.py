import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from .datasets import MNIST

# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
    ])

train_set = MNIST('Dataset', train=True, download=True, transform=transform)
val_set = MNIST('Dataset', train=False, transform=transform)