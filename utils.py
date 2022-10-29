import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import torchattacks
import numpy as np

get_model_name = lambda m: str(m.__class__).split('.')[-1][:-2]


# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
    ])

trainset = MNISTFast('Dataset', train=True, download=True, transform=transform)
valset = MNISTFast('Dataset', train=False, transform=transform)
# trainset = datasets.MNIST('Dataset', train=True, download=True, transform=transform)
# valset = datasets.MNIST('Dataset', train=False, transform=transform)

def uniform_perturb(x, eps = 0.1):
    grad = torch.rand_like(x)
    ub = torch.clamp(x + eps, min = 0, max = 1)
    lb = torch.clamp(x - eps, min = 0, max = 1)
    delta = ub - lb
    return delta * grad + lb  

import numpy as np

def probabilistically_robust_learning(model, loss_fn, xs, labels, k = 100, rho = 0.1):
    xs_k = xs.unsqueeze(0).repeat(k, 1, 1, 1, 1).view(k * xs.shape[0], *xs.shape[1:])
    xs_k += torch.randn_like(xs_k)
    labels_k = labels.unsqueeze(0).repeat(k, 1).view(k * labels.shape[0], *labels.shape[1:])

    scores_k = model(xs_k)

    losses = loss_fn(scores_k, labels_k)
    alpha = np.quantile(losses.detach().cpu(), 1 - rho)

    loss = F.threshold(losses, alpha, 0.).mean() / rho
    return loss, scores_k.view(k, scores_k.shape[0] // k, *scores_k.shape[1:]).mean(0)


        
        

def attack_model(model, loss_fn, batchSize, valset, num_epochs):
  

    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                                batch_size = batchSize,
                                                shuffle = False, num_workers=2) # No need.

    # GPU enabling.
    model = model.cuda()
    original_correct = 0.0
    original_cum_loss = 0.0

    attacked_correct = 0.0
    attacked_cum_loss = 0.0

    # Make a pass over the training data.
    model.eval()
    atk = torchattacks.PGD(model, eps=0.1, alpha=1/255, steps=num_epochs, random_start=False)
    
    for (i, (inputs, labels)) in enumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward pass. (Prediction stage)
        scores = model(inputs)
        loss = loss_fn(scores, labels)

        # Count how many correct in this batch.
        max_scores, max_labels = scores.max(1)
        original_correct += (max_labels == labels).sum().item()
        original_cum_loss += loss.item()
        
        
        # Attack the model using PGD
        adv_images = atk(inputs, labels)

        # Forward pass. (Prediction stage)
        scores = model(adv_images)
        loss = loss_fn(scores, labels)

        # Count how many correct in this batch.
        max_scores, max_labels = scores.max(1)
        attacked_correct += (max_labels == labels).sum().item()
        attacked_cum_loss += loss.item()

        original_accuracy = original_correct / len(valset)
        original_losses = original_cum_loss / (i + 1)

        attacked_accuracy = attacked_correct / len(valset)
        attacked_losses = attacked_cum_loss / (i + 1)      

    # print('Before. Avg-Loss: %.4f, Accuracy: %.4f' % 
    #         (original_cum_loss / (i + 1), original_correct / len(valset)))
    # print('After. Avg-Loss: %.4f, Accuracy: %.4f' % 
    #         (attacked_cum_loss / (i + 1), attacked_correct / len(valset)))
    
    return original_accuracy, original_losses, attacked_accuracy, attacked_losses, adv_images

class Ensemble(torch.nn.Module):
    def __init__(self, m1, m2):
        super(Ensemble, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        return (self.m1(x) + self.m2(x))/2

class DeltaEnsemble(torch.nn.Module):
    def __init__(self, m, eps = 0.1, n_neighb = 0):
        super(DeltaEnsemble, self).__init__()
        self.m = m
        self.eps = eps
        self.n_neighb = n_neighb

    def _get_neighb_steep(self, x, n_neighb):
        all_inputs = [x]
        for k in range(n_neighb):
            grad = torch.sigmoid(torch.rand_like(x).uniform_(-200, 200))
            ub = torch.clamp(x + self.eps, min = 0, max = 1)
            lb = torch.clamp(x - self.eps, min = 0, max = 1)
            delta = ub - lb
            x2 = delta * grad + lb
            all_inputs.append(x2)
        return torch.stack(all_inputs)

    def _cam_z(self, x):
        x_ = x.clone().detach()
        x_.requires_grad = True
        z_ = self.m._feature(x_)
        z = z_.detach()
        z = z +  torch.randn_like(z).normal_(std = z.std() / 10)
        loss_z = F.mse_loss(z_, z)
        loss_z.backward()
        return x_.grad

    def _cam_d(self, x):
        x_ = x.clone().detach()
        x_.requires_grad = True
        d_ = self.m(x_)
        d = torch.rand_like(d_).uniform_(-100, 100).softmax(-1)
        loss_d = F.kl_div(d_.log(), d)
        loss_d.backward()
        return x_.grad

    def _get_neighb_with_cam(self, x, n_neighb):
        cam_abs = self._cam_d(x).abs()
        cam_mask = cam_abs > np.percentile(cam_abs.cpu(), 75)

        x = x.unsqueeze(0)
        x_ = x.repeat(n_neighb, 1, 1, 1, 1)
        x_ = x + torch.randn_like(x_).sign() * self.eps * cam_mask
        x_ = torch.clamp(x_, min = 0, max = 1)
        x_ = torch.cat((x, x_), dim = 0)
        return x_

    def _get_neighb_uniform(self, x, n_neighb):
        x = x.unsqueeze(0)
        x_ = x.repeat(n_neighb, 1, 1, 1, 1)
        ub = torch.clamp(x + self.eps, min = 0, max = 1)
        lb = torch.clamp(x - self.eps, min = 0, max = 1)
        x_ = (ub - lb) * torch.rand_like(x_) + lb
        x_ = torch.cat((x, x_), dim = 0)
        return x_

    def _predict_neighb(self, x, n_neighb, batch_size = 10000):
        num_inputs = (n_neighb + 1) * len(x)
        all_inputs = self._get_neighb_uniform(x, n_neighb).view(num_inputs, *x.shape[1:])
        outputs = []
        for k in range(int(np.ceil(num_inputs / batch_size))):
            outputs.append(self.m(all_inputs[k * batch_size: (k + 1) * batch_size]))
        outputs = torch.cat(outputs, dim = 0).view((n_neighb + 1), len(x), -1)
        return outputs

    def forward(self, x, n_neighb = -1):
        if n_neighb == -1:
            n_neighb = self.n_neighb

        if n_neighb == 0:
            return self.m(x)
        else:            
            outputs = self._predict_neighb(x, n_neighb)
            return sum(outputs) / n_neighb