import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from utils.sampling import sample_uniform_linf_with_clamp, sample_uniform_l2

def probabilistically_robust_learning(model, loss_fn, xs, labels, k = 100, rho = 0.1):
    xs_k = xs.unsqueeze(0).repeat(k, 1, 1, 1, 1).view(k * xs.shape[0], *xs.shape[1:])
    xs_k += torch.randn_like(xs_k)
    labels_k = labels.unsqueeze(0).repeat(k, 1).view(k * labels.shape[0], *labels.shape[1:])

    scores_k = model(xs_k)

    losses = loss_fn(scores_k, labels_k)
    alpha = np.quantile(losses.detach().cpu(), 1 - rho)

    loss = F.threshold(losses, alpha, 0.).mean() / rho
    return loss, scores_k.view(k, scores_k.shape[0] // k, *scores_k.shape[1:]).mean(0)       
        


class Ensemble(torch.nn.Module):
    def __init__(self, m1, m2):
        super(Ensemble, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        return (self.m1(x) + self.m2(x))/2

class DeltaEnsemble(torch.nn.Module):
    def __init__(self, m, eps, n_neighb, batch_size):
        super(DeltaEnsemble, self).__init__()
        self.m = m
        self.eps = eps
        self.n_neighb = n_neighb
        self.batch_size = batch_size


    def _predict_neighb(self, x, n_neighb, batch_size = -1):
        if batch_size == -1:
            batch_size = self.batch_size
        num_inputs = (n_neighb + 1) * len(x)
        all_inputs = sample_uniform_linf_with_clamp(x, self.eps, n_neighb).view(num_inputs, *x.shape[1:])
        outputs = []
        for k in range(int(np.ceil(num_inputs / batch_size))):
            outputs.append(self.m(all_inputs[k * batch_size: (k + 1) * batch_size]))
        outputs = torch.cat(outputs, dim = 0).view((n_neighb + 1), len(x), -1)
        return outputs

    def forward(self, x, n_neighb = -1, batch_size = -1):
        if n_neighb == -1:
            n_neighb = self.n_neighb
        if batch_size == -1:
            batch_size = self.batch_size

        if n_neighb == 0:
            return self.m(x)
        else:            
            outputs = self._predict_neighb(x, n_neighb, batch_size)
            return sum(outputs) / n_neighb