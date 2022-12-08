import torch
import torch.nn as nn
from torch.nn import functional as F

from sampling import forward_samples


def ordinary_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def rand_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs + kw['noise_level'] * torch.randn_like(inputs))
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def augmented_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores_, inputs_ = forward_samples(net, inputs, **kw)
	loss = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'sum')

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()	
		correct_ = correct_.mean(dim = 0)
		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = (correct_ > kw['threshold']).sum().float()

	return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy}


def prl_step(net, batch, batch_idx, **kw):# dimension problem
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores_, inputs_ = forward_samples(net, inputs, **kw)
	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')

	alpha = torch.quantile(loss_.detach(), kw['threshold'], dim = 1)

	loss = F.relu(loss_ - alpha).mean(dim = 1).sum() / (1 - kw['threshold'])

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()	
		correct_ = correct_.mean(dim = 0)
		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = (correct_ > kw['threshold']).sum().float()

	return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy, 'alpha':alpha}


def our_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores_, inputs_ = forward_samples(net, inputs, **kw)

	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')

	sigma = torch.std(loss_[:, 1:], dim = 1)
	mu = torch.mean(loss_[:, 1:], dim = 1)

	loss = (mu + sigma).sum()

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()	
		correct_ = correct_.mean(dim = 0)
		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = (correct_ > kw['threshold']).sum().float()

	return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy, 'mu':mu.sum(), 'sigma':sigma.sum()}

def attacked_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def trades_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum') + F.kl_div(torch.log_softmax(scores, dim=1), F.softmax(net(inputs), dim=1), reduction='batchmean') * inputs.shape[0]

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def predict_step(net, batch, batch_idx, **kw):
	inputs, _ = batch
	inputs = inputs.to(kw['device'])
	scores = net(inputs)

	max_scores, max_labels = scores.max(1)
	return {'predictions':max_labels}

def perturbation_estimate_step(net, batch, batch_idx, **kw): # num = 40, batch_size = 10000
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	eps = torch.ones_like(labels).view(1, -1, 1, 1, 1) * 0.5

	for _ in range(num_estimates):
		scores_, inputs_ = forward_samples(net, inputs, **kw)
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float().mean(dim = 0).view(-1, 1, 1, 1)
		eps += (correct_ - 0.5)# * ((correct_ < 0.5).float() * 30 + 1)
		eps = torch.clamp(eps, lb, ub)

	return {'eps':eps.squeeze(), 'correct':correct_.squeeze()}#, 'samples':inputs_}
