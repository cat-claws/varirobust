import torch

def train(net, training_step, train_set, **kwargs):#, batch_size, optimizer, epoch, writer
	net = net.to(device)
	net.train()
	# Shuffling is needed in case dataset is not shuffled by default.
	train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, num_workers = 2, shuffle = True)

	outputs = []
	for batch_idx, batch in enumerate(train_loader):
		optimizer.zero_grad()
		output = training_step(net, batch, batch_idx, **kwargs)
		loss = output['loss']
		outputs.append(output)
		for k, v in output.items():
			writer.add_scalar("Step-" + k + "-train", v / batch_size, epoch * len(train_loader) + batch_idx)
		loss.backward()
		optimizer.step()

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/train", v / len(train_set), epoch)

	return net


def validate(net, validation_step, val_set, **kwargs):#, batch_size, epoch, writer
	net.eval()
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, num_workers = 2, shuffle = False)

	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = validation_step(net, batch, batch_idx, **kwargs)
			outputs.append(output)
			for k, v in output.items():
				writer.add_scalar("Step-" + k + "-valid", v / batch_size, epoch * len(val_loader) + batch_idx)

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/valid", v / len(val_set), epoch)


def attack(net, validation_step, attacked_step, val_set, **kwargs):#, batch_size, epoch, writer, torchattack, eps=0.1, alpha=1/255, steps=40, random_start=False
  
	net.eval()
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, num_workers = 2, shuffle = False)

	atk = Atk(net, eps=eps, alpha=alpha, steps=steps, random_start=random_start)

	outputs = []
	outputs_ = []
	for batch_idx, batch in enumerate(val_loader):
		with torch.no_grad():
			output = validation_step(net, batch, batch_idx, **kwargs)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})

		output_ = attacked_step(net, atk, batch, batch_idx, **kwargs)
		outputs_.append({k:v.detach().cpu() for k, v in output_.items()})

	outputs = {k: sum([dic[k] for dic in outputs]).item() for k in outputs[0]}
	outputs_ = {k: sum([dic[k] for dic in outputs_]).item() for k in outputs_[0]}
	for k, v in outputs.items():
		writer.add_scalar("Epoch-" + k + "/valid", v / len(val_set), epoch)
	for k, v in outputs_.items():
		writer.add_scalar("Epoch-" + k + "/attack", v / len(val_set), epoch)

	return outputs, outputs_


def predict(net, predict_step, val_set, **kwargs):# batch_size, epoch=None, writer=None
	net.eval()
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, num_workers = 2, shuffle = False)

	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = predict_step(net, batch, batch_idx, **kwargs)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})

	outputs = {k: torch.cat([dic[k] for dic in outputs], dim = 0).tolist() for k in outputs[0]} # array outputs

	return outputs