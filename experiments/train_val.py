import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

from utils import experiment, iterate
from mnist_models import ConvNet

m = ConvNet()
writer = SummaryWriter(comment=m._get_name() + '_train_val')

for epoch in range(100):
    m = iterate.train(m,
        iterate.mnist_rand_step,
        device = 'cuda',
        train_set = experiment.train_set,
        batch_size = 1000,
        optimizer = torch.optim.Adam(m.parameters(), lr = 0.001),
        epoch = epoch,
        writer = writer
    )

    iterate.validate(m,
        iterate.mnist_augmented_step,
        device = 'cuda',
        val_set = experiment.val_set,
        batch_size = 1000,
        epoch = epoch,
        writer = writer
    )

    iterate.attack(m,
        iterate.mnist_step,
        iterate.mnist_attacked_step,
        device = 'cuda',
        val_set = experiment.val_set,
        batch_size = 1000,
        epoch = epoch,
        writer = writer,
        torchattack=torchattacks.PGD,
        eps=0.1,
        alpha=1/255,
        steps=40,
        random_start=False
    )

    # iterate.attack(m,
    #     iterate.mnist_step,
    #     iterate.mnist_attacked_step,
    #     device = 'cuda',
    #     val_set = experiment.val_set,
    #     batch_size = 1000,
    #     epoch = epoch,
    #     writer = writer,
    #     torchattack=torchattacks.PGDL2,
    #     eps=0.5,
    #     alpha=0.2,
    #     steps=40,
    #     random_start=True
    # )

print(m)
torch.save(m.state_dict(), "checkpoints/" + m._get_name() + ".pt")

# outputs = iterate.predict(m,
#         iterate.mnist_delta_predict_step_linf,
#         device = 'cuda',
#         val_set = experiment.val_set,
#         batch_size = 100
# )

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()


