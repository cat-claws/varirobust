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
from utils import nets, datasets, iterate, misc, autonet

config = {
	'dataset':'CIFAR100',
	'training_step':'ordinary_step',
	# 'z':6,
	'model_name':'cifarwrn28_10_cifar100',
	# 'checkpoint':'checkpoints/ResNet18_cifar10_var_1000.pt',
	# 'initialization':'xavier_init',
	'batch_size':128,
	'optimizer':'SGD',
	'optimizer_config':{
		'lr':1e-1,
		'momentum':0.9,
		'weight_decay':1e-4,
	},
	# 'scheduler':'MultiStepLR',
	# 'scheduler_config':{
	# 	'milestones':[30, 60, 90, 120, 150],
	# 	'gamma':0.1
	# },
	'scheduler':'CosineAnnealingWarmRestarts',
	'scheduler_config':{
		'T_0':10,
		'T_mult':2,
		'eta_min':1e-6
	},
	# 'scheduler':'CyclicLR',
	# 'scheduler_config':{
	# 	'max_lr':0.1,
	# 	'base_lr':1e-5,
	# 	'step_size_up':20
	# },
	'sample_':'sample_uniform_linf_with_clamp',
	'num':100,	
	'eps':8/255,
	'attack':'PGD',
	'attack_config':{
		'eps':8/255,
		'alpha':1/255,
		'steps':20,
		'random_start':False,
	},
	# 'attack':'PGDL2',
	# 'attack_config':{
	# 	'eps':0.5, #PGD
	# 	'alpha':0.2,
	# 	'steps':40,
	# 	'random_start':True,
	# }
	'microbatch_size':200,
	'threshold':0.95,
	'adversarial':'TPGD',
	'adversarial_config':{
		'eps':8/255,
		'alpha':2/255,
		'steps':10,
	},
	'device':'cuda',
	'validation_step':'ordinary_step',
	'attacked_step':'attacked_step'
}

train_set, val_set, channel = misc.auto_sets(config['dataset'])
# m = nets.auto_net(channel).cuda()
m = autonet.load_model(config['model_name']).cuda()


# if 'checkpoint' in config:
# 	m.load_state_dict({k:v for k,v in torch.load(config['checkpoint']).items() if k in m.state_dict()})
# if 'initialization' in config:
# 	m.apply(vars(misc)[config['initialization']])

# for name, param in m.named_parameters():                
# 	if not (name.startswith('conv1.') or name.startswith('layer1.')):
# 		param.requires_grad = False

# import pytorchcv.model_provider
# m = pytorchcv.model_provider.get_model(f"resnet20_{config['dataset'].lower()}", pretrained=True).to(config['device'])
# m.features[0:2].apply(misc.weight_init)
# for name, param in m.named_parameters():                
# 	if not (name.startswith('features.init_block.') or name.startswith('features.stage1.')):
# 		param.requires_grad = False

writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['training_step']}", flush_secs=10)

import json
with open("pretrained/configs.json", 'a') as f:
	f.write(json.dumps({**{'run':writer.log_dir.split('/')[-1]}, **config}) + '\n')
	print(json.dumps(config, indent=4))

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = vars(steps)[v]
	elif k == 'sample_':
		config[k] = vars(sampling)[v]
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v](m.parameters(), **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])
	elif k == 'adversarial' or k == 'attack':
		config[k] = vars(torchattacks)[v](m, **config[k+'_config'])
		
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size =  config['batch_size'], num_workers = 2, shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size =  config['batch_size'], num_workers = 2, shuffle = False)

for epoch in range(200):
	if epoch > 0:
		iterate.train(m,
			train_loader = train_loader,
			epoch = epoch,
			writer = writer,
			atk = config['adversarial'],
			**config
		)

	iterate.validate(m,
		val_loader = val_loader,
		epoch = epoch,
		writer = writer,
		**config
	)

	# iterate.attack(m,
	# 	val_loader = val_loader,
	# 	epoch = epoch,
	# 	writer = writer,
	# 	atk = config['attack'],
	# 	**config
	# )

	torch.save(m.state_dict(), "checkpoints_/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(m)

outputs = iterate.predict(m,
	steps.predict_step,
	val_loader = val_loader,
	**config
)

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()
