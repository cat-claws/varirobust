import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from transformers.utils import ModelOutput
from scipy.stats import binomtest



def forward_with_cifar10_transform(net, x):
	T = transforms.Normalize(mean = torch.tensor([0.4914, 0.4822, 0.4465]), std = torch.tensor([0.2023, 0.1994, 0.2010]))
	return net(T(x))

def forward_in_microbatch(net, x, microbatch_size):
    logits = []
    for k in range(math.ceil(len(x) / microbatch_size)):
        logits.append(net(x[k * microbatch_size: (k + 1) * microbatch_size]))
    return torch.cat(logits, dim = 0)

def forward_samples(net, xs, microbatch_size):
	neighbour, batch_size = xs.shape[:2]	
	all_inputs = xs.view(-1, *xs.shape[2:])
	all_logits = forward_in_microbatch(net, all_inputs, microbatch_size).view(neighbour, batch_size, -1)
	return ModelOutput(
		logits = F.softmax(all_logits/1e-4, dim = -1).mean(0),# - outputs.mean(0).detach() + outputs.mean(0)
		all_logits = all_logits,
		all_inputs = all_inputs,
	)

def forward_with_sampling(net, x, microbatch_size, sample_, **kw):
	xs = sample_(x, **kw)
	return forward_samples(net, xs, microbatch_size)

def forward_with_certification(net, x, alpha, mu, pop, sample_, **kw):

	func = lambda k, n: bool(binomtest(k, n, p=mu, alternative='two-sided').pvalue < 2 * alpha)

	rejected = torch.zeros(len(x), dtype=bool, device=x.device)
	K = None

	while not rejected.all():
		x_ = x[~rejected]

		kw['num'] = pop // len(x_)
		outputs = forward_with_sampling(net, x_, pop, sample_, **kw)

		preds = F.softmax(outputs.all_logits/1e-4, dim = -1).sum(0)

		if K is None:
			K = preds
		else:
			K.masked_scatter_(~rejected.unsqueeze(1).expand_as(K), K[~rejected.unsqueeze(1).expand_as(K)] + preds.flatten())
		
		en = K.round().int().sum(1)
		ex = en - K.round().int().max(1)[0]

		rejected = torch.tensor(list(map(func, ex.tolist(), en.tolist())), device = rejected.device)

	return ModelOutput(
		logits = K,
		certified = ex/en < mu,
		mu = mu,
		alpha = alpha
	)
