import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '../adversarial-attacks-pytorch/')
sys.path.append('..')
sys.path.append('.')
import torchattacks

import steps
import sampling
from utils import datasets, iterate, misc, autonet

config = {
	'dataset':'SVHN',
	'model_name':'resnet18_cifar10_erm',
	'batch_size':8,
	'eps':8/255,
	# 'attack':'BruteForceUniform',
	# 'attack_config':{
	# 	'eps':8/255,
	# 	'alpha':1e-2,
	# 	'mu':1e-2,
	# 	'pop':256,
	# 	'verbose':False
	# },
	'attack':'PGD',
	'attack_config':{
		'eps':8/255,
		'alpha':0.003,
		'steps':20,
		'random_start':False,
	},
	'device':'cuda',
	'validation_step':'ordinary_step',
	'attacked_step':'attacked_step',
	# 'attacked_step':'binom_step',
}

train_set, val_set, _ = misc.auto_sets(config['dataset'])

m = autonet.load_model(config['model_name']).cuda()

# if 'model_name' in config:
# 	m.load_state_dict({k:v for k,v in torch.load(config['model_name']).items() if k in m.state_dict()})

writer = SummaryWriter(comment = f"_{config['dataset']}_{config['model_name'].split('/')[-1]}", flush_secs=10)


for k, v in config.items():
	if k.endswith('_step'):
		config[k] = vars(steps)[v]
	elif k == 'adversarial' or k == 'attack':
		config[k] = vars(torchattacks)[v](m, **config[k+'_config'])
		
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size =  config['batch_size'], num_workers = 2, shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size =  config['batch_size'], num_workers = 2, shuffle = False)

for epoch in range(504, 716):
	m.load_state_dict(torch.load(f'pretrained/resnet18_svhn_var_{epoch}.pt'))
# 	if epoch > 0:
# 		iterate.train(m,
# 			train_loader = train_loader,
# 			epoch = epoch,
# 			writer = writer,
# 			atk = config['adversarial'],
# 			**config
# 		)

	# iterate.validate(m,
	# 	val_loader = val_loader,
	# 	epoch = epoch,
	# 	writer = writer,
	# 	# atk = config['attack'],
	# 	**config
	# )

	iterate.attack(m,
		val_loader = val_loader,
		epoch = epoch,
		writer = writer,
		atk = config['attack'],
		**config
	)

# 	torch.save(m.state_dict(), "model_names_/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")


	# outputs = iterate.attack(m,
	# 	val_loader = val_loader,
	# 	epoch = epoch,
	# 	writer = writer,
	# 	atk = config['attack'],
	# 	**config
	# )

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()
