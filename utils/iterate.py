import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import metrics

writer = SummaryWriter()

def weight_reset(m):
	reset_parameters = getattr(m, "reset_parameters", None)
	if callable(reset_parameters):
		m.reset_parameters()


def mnist_step(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	scores = net(inputs)
	loss = F.nll_loss(torch.log(scores + 1e-9), labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def mnist_rand_step(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	scores = net(inputs + 0.6 * torch.randn_like(inputs)) # add gaussian noise
	loss = F.nll_loss(torch.log(scores + 1e-9), labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def mnist_augmented_step(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	
	scores = net(inputs)
	loss = F.nll_loss(torch.log(scores + 1e-9), labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()

	aug_acc, beta_quant_acc = metrics.augmented_accuracy(net, inputs, labels, eps = 0.1, n_samples = 100, beta = 0.1)
	return {'loss':loss, 'correct':correct, 'quantile':beta_quant_acc, 'augmented':aug_acc}


def mnist_attacked_step(net, attack, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	inputs_ = attack(inputs, labels)

	scores = net(inputs_)
	loss = F.nll_loss(torch.log(scores + 1e-9), labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def mnist_predict_step(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, _ = batch
	inputs = inputs.to(device)
	scores = net(inputs)

	max_scores, max_labels = scores.max(1)
	return {'predictions':max_labels}

def mnist_delta_predict_step_linf(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	M = 200
	eps = torch.ones_like(labels) * 0.5

	for _ in range(50):
		inputs_, labels_ = inputs.repeat(M, 1, 1, 1), labels.repeat(M)
		eps_ = eps.view(-1, 1, 1, 1).repeat(M, *inputs_.shape[1:])
		inputs_ = torch.clamp(inputs_ + 2 * eps_ * torch.rand_like(inputs_) - eps_, 0, 1.)

		scores_ = net(inputs_)
		_, max_labels_ = scores_.max(1)
		correct_ = (max_labels_ == labels_).view(M, -1).float().mean(dim = 0)
		eps += (correct_ - 0.5) * ((correct_ < 0.5).float() * 30 + 1)
		eps = torch.clamp(eps, 0, 1)

	return {'predictions':eps, 'correct':correct_, 'samples':inputs_}

def mnist_delta_predict_step_l2(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	M = 40
	eps = torch.zeros_like(labels).float()

	for _ in range(50):
		inputs_, labels_ = inputs.repeat(M, 1, 1, 1), labels.repeat(M)
		eps_ = eps.view(-1, 1, 1, 1).repeat(M, *inputs_.shape[1:])

		grad = torch.randn_like(inputs_)
		grad_norms = torch.norm(grad.view(inputs_.shape[0], -1), p=2, dim=1) + 1e-7
		grad = grad / grad_norms.view(inputs_.shape[0], 1, 1, 1)

		factor = torch.rand_like(grad) ** (1 / torch.numel(grad[0]))
		delta = eps_ * grad * factor
		inputs_ = torch.clamp(inputs_ + delta, min=0, max=1)

		scores_ = net(inputs_)
		_, max_labels_ = scores_.max(1)
		correct_ = (max_labels_ == labels_).view(M, -1).float().mean(dim = 0)
		eps += (correct_ - 0.5) * ((correct_ < 0.5).float() * 30 + 1)
		eps = torch.clamp(eps, 0, 20)

	return {'predictions':eps, 'correct':correct_, 'samples':inputs_}

def train(model, training_step, device, train_set, batch_size, optimizer, epoch, writer):
	model = model.to(device)
	model.train()
	# Shuffling is needed in case dataset is not shuffled by default.
	train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)

	outputs = []
	for batch_idx, batch in enumerate(train_loader):
		optimizer.zero_grad()
		output = training_step(model, batch, batch_idx, device)
		loss = output['loss']
		outputs.append(output)
		for k, v in output.items():
			writer.add_scalar("Step-" + k + "-train", v / batch_size, epoch * len(train_loader) + batch_idx)
		loss.backward()
		optimizer.step()

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/train", v / len(train_set), epoch)

	return model


def validate(model, validation_step, device, val_set, batch_size, epoch, writer):
	model.eval()
	# We don't need to bach the validation set but let's do it anyway.
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, shuffle = False) # No need.

	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = validation_step(model, batch, batch_idx, device)
			outputs.append(output)
			for k, v in output.items():
				writer.add_scalar("Step-" + k + "-valid", v / batch_size, epoch * len(val_loader) + batch_idx)

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/valid", v / len(val_set), epoch)


def attack(model, validation_step, attacked_step, device, val_set, batch_size, epoch, writer, torchattack, eps=0.1, alpha=1/255, steps=40, random_start=False):
  
	model.eval()
	# We don't need to bach the validation set but let's do it anyway.
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, shuffle = False) # No need.

	atk = torchattack(model, eps=eps, alpha=alpha, steps=steps, random_start=random_start)

	outputs = []
	outputs_ = []
	for batch_idx, batch in enumerate(val_loader):
		with torch.no_grad():
			output = validation_step(model, batch, batch_idx, device)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})

		output_ = attacked_step(model, atk, batch, batch_idx, device)
		outputs_.append({k:v.detach().cpu() for k, v in output_.items()})

	outputs = {k: sum([dic[k] for dic in outputs]).item() for k in outputs[0]}
	outputs_ = {k: sum([dic[k] for dic in outputs_]).item() for k in outputs_[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/valid", v / len(val_set), epoch)
	for k, v in outputs_.items():
		writer.add_scalar("Epoch-" + k + "/attack", v / len(val_set), epoch)

	return outputs, outputs_

def predict(model, predict_step, device, val_set, batch_size, epoch=None, writer=None):
	model.eval()
	# We don't need to bach the validation set but let's do it anyway.
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, shuffle = False) # No need.

	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = predict_step(model, batch, batch_idx, device)
			outputs.append(output)
	# 		for k, v in output.items():
	# 			writer.add_scalar("Step-" + k + "-valid", v / batch_size, epoch * len(val_loader) + batch_idx)

	outputs = {k: torch.cat([dic[k] for dic in outputs], dim = 0).tolist() for k in outputs[0]} # array outputs
	# for k, v in outputs.items():
	# 	writer.add_scalar("Epoch-" + k + "/valid", v / len(val_set), epoch)
	return outputs
