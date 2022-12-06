import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

import steps
from sampling import sample_uniform_linf_with_clamp
from utils import nets, datasets, iterate, misc


# datasets.auto_set('SVHN', download=True, split = 'train', transform=misc.transforms_3)
m = nets.auto_net(1).cuda()
writer = SummaryWriter(comment=m._get_name() + '_train_val')

for epoch in range(1):
    m = iterate.train(m,
        steps.trades_step,
        noise_level = 0.6,
        sample_ = sample_uniform_linf_with_clamp,
        num = 2,
        eps = 0.1,
        microbatch_size = 10000,
        threshold = 0.9,
        device = 'cuda',
        train_set = datasets.auto_set('MNIST', download=True, train = False, transform=misc.transforms_1),
        batch_size = 1000,
        optimizer = torch.optim.Adam(m.parameters(), lr = 0.001),
        epoch = epoch,
        writer = writer,
        atk = torchattacks.TPGD(m, eps=8/255, alpha=2/255, steps=10)
    )

    # iterate.validate(m,
    #     steps.auto_step,
    #     device = 'cuda',
    #     val_set = datasets.auto_set('MNIST', download=True, train = False, transform=misc.transforms_1),
    #     batch_size = 1000,
    #     epoch = epoch,
    #     writer = writer
    # )

    # iterate.attack(m,
    #     steps.ordinary_step,
    #     steps.auto_step,
    #     device = 'cuda',
    #     val_set = datasets.auto_set('MNIST', download=True, train = False, transform=misc.transforms_1),
    #     batch_size = 1000,
    #     epoch = epoch,
    #     writer = writer,
    #     atk=torchattacks.PGD(m, eps=0.1, alpha=1/255, steps=40, random_start=False),
    # )

    # iterate.attack(m,
    #     steps.ordinary_step,
    #     steps.auto_step,
    #     device = 'cuda',
    #     val_set = datasets.auto_set('MNIST', download=True, train = False, transform=misc.transforms_1),
    #     batch_size = 1000,
    #     epoch = epoch,
    #     writer = writer,
    #     atk=torchattacks.PGDL2(m, eps=0.5, alpha=0.2, steps=40, random_start=True),
    # )


print(m)
torch.save(m.state_dict(), "checkpoints/" + m._get_name() + ".pt")

outputs = iterate.predict(m,
        steps.auto_step,
        device = 'cuda',
        val_set = datasets.auto_set('MNIST', download=True, train = False, transform=misc.transforms_1),
        batch_size = 100,
        predict = True
)

print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()


