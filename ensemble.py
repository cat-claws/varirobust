import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from sampling import forward_samples




class Ensemble(torch.nn.Module):
	def __init__(self, m1, m2):
		super(Ensemble, self).__init__()
		self.m1 = m1
		self.m2 = m2

	def forward(self, x):
		return (self.m1(x) + self.m2(x))/2

class SampleEnsemble(torch.nn.Module):
	def __init__(self, m, **kw):
		super(SampleEnsemble, self).__init__()
		self.m = m
		self.kw = forward_samples(m, x, **kw)

	def forward(self, x, n_neighb = -1, batch_size = -1):
		forward_samples(self.m, self.sampling, x, self.eps, n_neighb, batch_size)
			return sum(outputs) / n_neighb
