import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from utils.misc import forward_micro

def sample_uniform_linf(x, eps, num):
    x_ = x.repeat(num, 1, 1, 1, 1)
    ub = x + eps
    lb = x - eps
    x_ = (ub - lb) * torch.rand_like(x_) + lb
    x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
    return x_

def sample_uniform_linf_with_clamp(x, eps, num):
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

def forward_samples(net, x, **kw):
    num_inputs = (kw['num'] + 1) * len(x)
    all_inputs = kw['sample_'](x, kw['eps'], kw['num'])
    outputs = forward_micro(net, all_inputs, kw['microbatch_size'])
    return outputs, all_inputs
