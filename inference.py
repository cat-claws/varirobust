import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import numpy as np
from sampling import forward_samples


def ci(net, x, **kw):
	outputs, _ = forward_samples(net, x, **kw)
	return F.softmax(outputs/1e-4, dim = -1).mean(0)# - outputs.mean(0).detach() + outputs.mean(0)

class CI(nn.Module):
	def __init__(self, net, **kw):
		super(CI, self).__init__()
		self.net = net
		self.kw = kw
		
	def forward(self, x):
		return ci(self.net, x, **self.kw)


def predict_with_transform(net, x):
    T = transforms.Normalize(mean = torch.tensor([0.4914, 0.4822, 0.4465]), std = torch.tensor([0.2023, 0.1994, 0.2010]))
    return net(T(x))

class LambdaI(nn.Module):
	def __init__(self, net, predict=predict_with_transform):
		super(LambdaI, self).__init__()
		self.net = net
		self.predict = predict
		
	def forward(self, x):
		return self.predict(self.net, x)