import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

import steps
import sampling
from utils import nets, datasets, iterate, misc

config = {
	'dataset':'MNIST',
	'training_step':'our_step',
	'batch_size':64,
	# 'noise_level':0.6,
	'sample_':'sample_uniform_linf_with_clamp',
	'num':50,
	'eps':0.3,
	'microbatch_size':10000,
	'threshold':0.9,
	'device':'cuda',
	'validation_step':'augmented_step',
	'attacked_step':'attacked_step'
	}

train_set, val_set, channel = misc.auto_sets(config['dataset'])
m = nets.auto_net(channel).cuda()

writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['training_step']}")

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = vars(steps)[v]
	elif k == 'sample_':
		config[k] = vars(sampling)[v]


for epoch in range(300):
	m = iterate.train(m,
		train_set = train_set,
		optimizer = torch.optim.Adam(m.parameters(), lr = 0.001),
		epoch = epoch,
		writer = writer,
		# atk = torchattacks.TPGD(m, eps=8/255, alpha=2/255, steps=10)
		**config
	)

	iterate.attack(m,
		val_set = val_set,
		epoch = epoch,
		writer = writer,
		atk = torchattacks.PGD(m, eps=0.3, alpha=0.1, steps=10, random_start=False),
	#     atk = torchattacks.PGDL2(m, eps=0.5, alpha=0.2, steps=40, random_start=True),
		**config
	)


print(m)
torch.save(m.state_dict(), "checkpoints/" + writer.log_dir.split('/')[-1] + ".pt")

outputs = iterate.predict(m,
        steps.predict_step,
        val_set = val_set,
)

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()