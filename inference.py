import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from sampling import forward_samples




# class CI(torch.nn.Module):
# 	def __init__(self, m1, m2):
# 		super(Ensemble, self).__init__()
# 		self.m1 = m1
# 		self.m2 = m2

# 	def forward(self, x):
# 		return (self.m1(x) + self.m2(x))/2

def ci(net, **kw)
	outputs, _ = forward_samples(net, x, **kw)
	return torch.nn.functional.normalize(outputs, p=1, dim=1).mean(dim = 0)
