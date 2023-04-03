import torch
import torch.nn as nn
from torch.nn import functional as F

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
		