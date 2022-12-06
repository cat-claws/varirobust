import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.sampling import sample_uniform_linf_with_clamp, sample_uniform_l2, forward_samples


def auto_step(net, batch, batch_idx, **kwargs):
	if 'noise_level' in kwargs:
		return rand_step(net, batch, batch_idx, **kwargs)
	elif 'num_estimates' in kwargs:
		return perturbation_estimate_step(net, batch, batch_idx, **kwargs)
	elif 'num' in kwargs:
		return augmented_step(net, batch, batch_idx, **kwargs)
	elif 'atk' in kwargs:
        if kwargs['atk'] == 
		return attacked_step(net, batch, batch_idx, **kwargs)
	elif 'predict' in kwargs:
		return predict_step(net, batch, batch_idx, **kwargs)
	else:
		return ordinary_step(net, batch, batch_idx, **kwargs)


def ordinary_step(net, batch, batch_idx, **kwargs):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def rand_step(net, batch, batch_idx, **kwargs):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	scores = net(inputs + noise_level * torch.randn_like(inputs)) # add gaussian noise default 0.6
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def augmented_step(net, batch, batch_idx, **kwargs): # eps = 0.1, n_samples = 100, beta = 0.1
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()

	with torch.no_grad():
        scores_, inputs_ = forward_samples(net, inputs, **kwargs)
        _, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float().mean(dim = 0)
        
        augmented_accuracy = correct_.sum()
        quantile_accuracy = (correct_ > threshold).sum().float()

    return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy}


def attacked_step(net, batch, batch_idx, **kwargs):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	inputs_ = atk(inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def predict_step(net, batch, batch_idx, **kwargs):
	inputs, _ = batch
	inputs = inputs.to(device)
	scores = net(inputs)

	max_scores, max_labels = scores.max(1)
	return {'predictions':max_labels}

def perturbation_estimate_step(net, batch, batch_idx, **kwargs): # num = 40, batch_size = 10000
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)

	eps = torch.ones_like(labels).view(1, -1, 1, 1, 1) * 0.5

	for _ in range(num_estimates):
		scores_, inputs_ = forward_samples(net, inputs, **kwargs)
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float().mean(dim = 0).view(-1, 1, 1, 1)
		eps += (correct_ - 0.5)# * ((correct_ < 0.5).float() * 30 + 1)
		eps = torch.clamp(eps, lb, ub)

	return {'eps':eps.squeeze(), 'correct':correct_.squeeze()}#, 'samples':inputs_}


def trades_step(net, batch, batch_idx, **kwargs):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	inputs_ = atk(inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum') + F.kl_div(torch.log_softmax(scores, dim=1), net(inputs), reduction='batchmean') * inputs.shape[0]

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}