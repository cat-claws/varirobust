import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

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

# def _cam_z(self, x):
#     x_ = x.clone().detach()
#     x_.requires_grad = True
#     z_ = self.m._feature(x_)
#     z = z_.detach()
#     z = z +  torch.randn_like(z).normal_(std = z.std() / 10)
#     loss_z = F.mse_loss(z_, z)
#     loss_z.backward()
#     return x_.grad

# def _cam_d(self, x):
#     x_ = x.clone().detach()
#     x_.requires_grad = True
#     d_ = self.m(x_)
#     d = torch.rand_like(d_).uniform_(-100, 100).softmax(-1)
#     loss_d = F.kl_div(d_.log(), d)
#     loss_d.backward()
#     return x_.grad

# def _get_neighb_with_cam(self, x, num):
#     cam_abs = self._cam_d(x).abs()
#     cam_mask = cam_abs > np.percentile(cam_abs.cpu(), 75)

#     x = x.unsqueeze(0)
#     x_ = x.repeat(num, 1, 1, 1, 1)
#     x_ = x + torch.randn_like(x_).sign() * self.eps * cam_mask
#     x_ = torch.clamp(x_, min = 0, max = 1)
#     x_ = torch.cat((x.repeat(1, 1, 1, 1, 1), x_), dim = 0)
#     return x_

# def _get_neighb_steep(self, x, n_neighb):
#     all_inputs = [x]
#     for k in range(n_neighb):
#         grad = torch.sigmoid(torch.rand_like(x).uniform_(-200, 200))
#         ub = torch.clamp(x + self.eps, min = 0, max = 1)
#         lb = torch.clamp(x - self.eps, min = 0, max = 1)
#         delta = ub - lb
#         x2 = delta * grad + lb
#         all_inputs.append(x2)
#     return torch.stack(all_inputs)

# def _cam_z(self, x):
#     x_ = x.clone().detach()
#     x_.requires_grad = True
#     z_ = self.m._feature(x_)
#     z = z_.detach()
#     z = z +  torch.randn_like(z).normal_(std = z.std() / 10)
#     loss_z = F.mse_loss(z_, z)
#     loss_z.backward()
#     return x_.grad

# def _cam_d(self, x):
#     x_ = x.clone().detach()
#     x_.requires_grad = True
#     d_ = self.m(x_)
#     d = torch.rand_like(d_).uniform_(-100, 100).softmax(-1)
#     loss_d = F.kl_div(d_.log(), d)
#     loss_d.backward()
#     return x_.grad

# def _get_neighb_with_cam(self, x, n_neighb):
#     cam_abs = self._cam_d(x).abs()
#     cam_mask = cam_abs > np.percentile(cam_abs.cpu(), 75)

#     x = x.unsqueeze(0)
#     x_ = x.repeat(n_neighb, 1, 1, 1, 1)
#     x_ = x + torch.randn_like(x_).sign() * self.eps * cam_mask
#     x_ = torch.clamp(x_, min = 0, max = 1)
#     x_ = torch.cat((x, x_), dim = 0)
#     return x_

# def _get_neighb_uniform(self, x, n_neighb):
#     x = x.unsqueeze(0)
#     x_ = x.repeat(n_neighb, 1, 1, 1, 1)
#     ub = torch.clamp(x + self.eps, min = 0, max = 1)
#     lb = torch.clamp(x - self.eps, min = 0, max = 1)
#     x_ = (ub - lb) * torch.rand_like(x_) + lb
#     x_ = torch.cat((x, x_), dim = 0)
#     return x_
