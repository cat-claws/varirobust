"""
This is pretty hard-coded, and only serves to simplifying load different models.
All models are loaded to take image inputs bounded by [0, 1.], instead of [0, 255].
Model outputs are usually logits, but sometimes it could be in huggingface ModelOutput.
"""

import torch

from .nets import ConvNet, CNN7, ResNet18, ResNet50
from .convmed import ConvMed, ConvMedBig
from .wide_resnet_bn import wide_resnet_8


class LambdaNet(torch.nn.Module):
	def __init__(self, net, forward, **kw):
		super(LambdaNet, self).__init__()
		self.net = net
		self._forward = forward
		self.kw = kw
		self.logits_only = False
		
	def forward(self, x):
		outputs = self._forward(self.net, x, **self.kw)
		if self.logits_only:
			return outputs.logits
		else:
			return outputs


def load_model(model_name):
    if model_name == 'convnet_mnist':
        return ConvNet()

    elif model_name.startswith('convnet_mnist_'):
        model = ConvNet()
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'resnet18_cifar10' or model_name == 'resnet18_svhn':
        return ResNet18()

    elif model_name.startswith('resnet18_'):
        model = ResNet18()
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model