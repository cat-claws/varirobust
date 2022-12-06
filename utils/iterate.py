import torch

def train(net, training_step, train_set, optimizer, **kw):
	net = net.to(kw['device'])
	net.train()
	# Shuffling is needed in case dataset is not shuffled by default.
	train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size =  kw['batch_size'], num_workers = 2, shuffle = True)

	outputs = []
	for batch_idx, batch in enumerate(train_loader):
		optimizer.zero_grad()
		output = training_step(net, batch, batch_idx, **kw)
		loss = output['loss']
		outputs.append(output)
		for k, v in output.items():
			kw['writer'].add_scalar("Step-" + k + "-train", v / kw['batch_size'], kw['epoch'] * len(train_loader) + batch_idx)
		loss.backward()
		optimizer.step()

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/train", v / len(train_set), kw['epoch'])

	return net


def validate(net, validation_step, val_set, **kw):
	net.eval()
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size =  kw['batch_size'], num_workers = 2, shuffle = False)

	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = validation_step(net, batch, batch_idx, **kw)
			outputs.append(output)
			for k, v in output.items():
				kw['writer'].add_scalar("Step-" + k + "-valid", v / kw['batch_size'], kw['epoch'] * len(val_loader) + batch_idx)

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/valid", v / len(val_set), kw['epoch'])


def attack(net, validation_step, attacked_step, val_set, **kw):
  
	net.eval()
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size =  kw['batch_size'], num_workers = 2, shuffle = False)

	outputs = []
	outputs_ = []
	for batch_idx, batch in enumerate(val_loader):
		with torch.no_grad():
			output = validation_step(net, batch, batch_idx, **kw)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})

		output_ = attacked_step(net, batch, batch_idx, **kw)
		outputs_.append({k:v.detach().cpu() for k, v in output_.items()})

	outputs = {k: sum([dic[k] for dic in outputs]).item() for k in outputs[0]}
	outputs_ = {k: sum([dic[k] for dic in outputs_]).item() for k in outputs_[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/valid", v / len(val_set), kw['epoch'])
	for k, v in outputs_.items():
		kw['writer'].add_scalar("Epoch-" + k + "/attack", v / len(val_set), kw['epoch'])

	return outputs, outputs_


def predict(net, predict_step, val_set, **kw):
	net.eval()
	val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size =  kw['batch_size'], num_workers = 2, shuffle = False)

	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = predict_step(net, batch, batch_idx, **kw)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})

	outputs = {k: torch.cat([dic[k] for dic in outputs], dim = 0).tolist() for k in outputs[0]} # array outputs

	return outputs