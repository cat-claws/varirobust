import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np
# import torchattacks
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
		
def mnist_step(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	scores = net(inputs)
	loss = F.nll_loss(torch.log(scores + 1e-9), labels)

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def mnist_rand_step(net, batch, batch_idx, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	inputs, labels = batch
	inputs, labels = inputs.to(device), labels.to(device)
	scores = net(inputs + 0.6 * torch.randn_like(inputs)) # add gaussian noise
	loss = F.nll_loss(torch.log(scores + 1e-9), labels)

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).mean()
	return {'loss':loss, 'accuracy':correct}


def train(model, training_step, device, train_set, batch_size, optimizer, epoch, writer):
	model = model.to(device)
    model.train()
	# Shuffling is needed in case dataset is not shuffled by default.
	train_loader = torch.utils.data.DataLoader(dataset = trainset,
												batch_size = batchSize,
												shuffle = True)

	outputs = []
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = training_step(model, batch_idx, batch, device)
		loss = output['loss']
		outputs.append(output)
		for k, v in outputs.items():
			writer.add_scalar("Step-" + k + "-train", v, epoch * len(train_loader) + batch_idx)
        loss.backward()
        optimizer.step()

	outputs = {k: torch.cat([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/train", sum(v) / len(train_set), epoch)


def validate(model, validation_step, device, val_set, batch_size, epoch, writer):
    model.eval()
	# We don't need to bach the validation set but let's do it anyway.
	val_loader = torch.utils.data.DataLoader(dataset = valset,
												batch_size = batchSize,
												shuffle = False) # No need.

	outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            output = training_step(model, batch_idx, batch, device)
            loss = output['loss']
			outputs.append(output)
			for k, v in outputs.items():
				writer.add_scalar("Step-" + k + "-valid", v, epoch * len(val_loader) + batch_idx)

	outputs = {k: torch.cat([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/valid", sum(v) / len(val_set), epoch)  

	
