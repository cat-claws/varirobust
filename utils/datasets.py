import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

class MNIST(datasets.MNIST):
	# fast version of torchvision.datasets.MNIST
	def __init__(
		self,
		root: str,
		train: bool = True,
		transform = None,
		target_transform = None,
		download: bool = False,
	) -> None:
		super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
		self.data_ = []
		for index in range(len(self.data)):
			img, target = super().__getitem__(index)
			self.data_.append((img.clone().detach(), target))
			
	def __getitem__(self, index: int):
		return self.data_[index]


