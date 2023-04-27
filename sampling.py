import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T


# import numpy as np

def sample_uniform_linf(x, eps, num):
	x_ = x.repeat(num, 1, 1, 1, 1)
	ub = x + eps
	lb = x - eps
	x_ = (ub - lb) * torch.rand_like(x_) + lb
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_

def sample_uniform_linf_with_clamp(x, eps, num):
	x_ = sample_uniform_linf(x, eps, num)
	x_ = torch.clamp(x_, min = 0, max = 1)
	return x_

def sample_uniform_linf_with_soft_clamp(x, eps, num):
	x_ = x.repeat(num, 1, 1, 1, 1)
	ub = torch.clamp(x + eps, min = 0, max = 1)
	lb = torch.clamp(x - eps, min = 0, max = 1)
	x_ = (ub - lb) * torch.rand_like(x_) + lb
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_

def sample_uniform_l2(x, eps, num):
	x_ = x.repeat(num, 1, 1, 1, 1)
	u = torch.randn_like(x_)
	norm = torch.norm(u, dim = (-2, -1), p = 2, keepdim = True)
	norm = (norm ** 2 + torch.randn_like(norm) ** 2 + torch.randn_like(norm) ** 2) ** 0.5
	x_ = x + u / norm * eps
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_
    

def sample_uniform_l2_with_clamp(x, eps, num):
	x_ = sample_uniform_l2(x, eps, num)
	x_ = torch.clamp(x_, min = 0, max = 1)
	return x_
    
def sample_steep(x, eps, num):
	all_inputs = [x]
	for k in range(num):
		grad = torch.sigmoid(torch.rand_like(x).uniform_(-200, 200))
		ub = torch.clamp(x + eps, min = 0, max = 1)
		lb = torch.clamp(x - eps, min = 0, max = 1)
		delta = ub - lb
		x2 = delta * grad + lb
		all_inputs.append(x2)
	return torch.stack(all_inputs)

def sample_random_rotation(x, eps, num):

	x_ = torch.stack(list(map(
		T.RandomRotation(degrees=eps),
		images.repeat(num, 1, 1, 1)
	)))

	x_ = x_.view(num, *x.shape)
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_

def sample_random_translation(x, eps, num):

	x_ = torch.stack(list(map(
		T.RandomAffine(degrees=0, translate=eps),
		images.repeat(num, 1, 1, 1)
	)))
	
	x_= x_.view(num, *x.shape)
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_

def sample_random_affine(x, eps, num):

	x_ = torch.stack(list(map(
		T.RandomAffine(degrees=eps.pop(), translate=eps),
		images.repeat(num, 1, 1, 1)
	)))

	x_ = x_.view(num, *x.shape)
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_

def sample_random_scale(x, eps, num):

	x_ = torch.stack(list(map(
		T.RandomResizedCrop(size=x.shape[-1], scale=eps, ratio=(1, 1), antialias = True),
		images.repeat(num, 1, 1, 1)
	)))

	x_ = x_.view(num, *x.shape)
	x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
	return x_


# from statsmodels.stats import weightstats

# def ztest(x, threshold, alpha):
# 	x = x.cpu().numpy()
# 	_, pvalue = weightstats.ztest(x1=x, x2=None, value=threshold, alternative='larger')
# 	return torch.from_numpy(pvalue < alpha).float()

# def sprt(x, threshold, alpha):
# 	x = x.cpu().numpy()
# 	n = x.shape[0]
# 	m = n - x.sum(0) + 1e-9
# 	pr = ((threshold+0.02)**m * (1-threshold-0.02)**(n-m))/((threshold-0.02)**m * (1-threshold+0.02)**(n-m) + 1e-9)

# 	h0 = (1 - alpha) / alpha
# 	h1 = alpha / (1 - alpha)

# 	cert = np.logical_or(pr > h0, pr < h1).sum()
# 	if cert < x.shape[1]:
# 		print('Cannot Accept H0. p < {} or H1. p > {} after {}/{} tests.'.format(h0, h1, cert, n))
# 	else:
# 		print(cert)

# 	return torch.from_numpy(pr <= h1).float()
