import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

from utils import experiment, iterate
from mnist_models import ConvNet

m = ConvNet()
m = m.to('cuda' if torch.cuda.is_available() else 'cpu')

for ckpt in ['checkpoints/ConvNet.pt', 'checkpoints/ConvNet_TRADES.pt', 'checkpoints/ConvNet_CVaR.pt']:

    writer = SummaryWriter(comment=ckpt)
    m.load_state_dict({k:torch.load(ckpt)[k] for k in m.state_dict()})


    iterate.attack(m,
            iterate.mnist_step,
            iterate.mnist_attacked_step,
            device = 'cuda',
            val_set = experiment.val_set,
            batch_size = 1000,
            epoch = 0,
            writer = writer,
            torchattack=torchattacks.PGD,
            eps=0.3,
            alpha=1/255,
            steps=40,
            random_start=False
        )

    import numpy as np
    from ensemble import DeltaEnsemble


    ns_neighb = range(10)
    epsilons = np.linspace(0.1, 0.5, 9)
    original_accuracy = np.empty((len(ns_neighb), len(epsilons)))
    attacked_accuracy = np.empty((len(ns_neighb), len(epsilons)))

    for i, n_neighb in enumerate(ns_neighb):
        for j, eps in enumerate(epsilons):
            m_ = DeltaEnsemble(m, n_neighb = n_neighb, eps = eps, batch_size = 2000)
            m_.eval()

            outputs, outputs_ = iterate.attack(m_,
                iterate.mnist_step,
                iterate.mnist_attacked_step,
                device = 'cuda',
                val_set = experiment.val_set,
                batch_size = 500,
                epoch = 1,
                writer = writer,
                torchattack=torchattacks.PGD,
                eps=0.3,
                alpha=1/255,
                steps=40,
                random_start=False
            )

            original_acc, attacked_acc = outputs['correct'] / len(experiment.val_set), outputs_['correct'] / len(experiment.val_set)
            original_accuracy[i, j] = original_acc
            attacked_accuracy[i, j] = attacked_acc

    writer.add_image('original accuracy', experiment.plot_to_image(experiment.plot_trend(original_accuracy, 'original accuracy', ns_neighb, epsilons, 'num of neighbours', 'size of delta')), 1)
    writer.add_image('attacked accuracy', experiment.plot_to_image(experiment.plot_trend(attacked_accuracy, 'attacked accuracy', ns_neighb, epsilons, 'num of neighbours', 'size of delta')), 1)
    writer.flush()
    writer.close()