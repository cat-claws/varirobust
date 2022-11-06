import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

from utils import experiment, iterate
from mnist_models import ConvNet

m = ConvNet()
m = m.to('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
m.load_state_dict(torch.load('checkpoints/ConvNet.pt'))
# m.load_state_dict(torch.load('checkpoints/ConvNet_TRADES.pt'))
# m.load_state_dict(torch.load('checkpoints/ConvNet_CVaR.pt'))


print('baseline: ')
# utils.attack_model(m, loss_fn, 1000, utils.valset, 400)
iterate.attack(m,
        iterate.mnist_step,
        iterate.mnist_attacked_step,
        device = 'cuda',
        val_set = experiment.val_set,
        batch_size = 1000,
        epoch = 0,
        writer = writer,
        torchattack=torchattacks.PGD,
        eps=0.1,
        alpha=1/255,
        steps=40,
        random_start=False
    )

import numpy as np
from ensemble import DeltaEnsemble
# original_accuracy = {}
# attacked_accuracy = {}


ns_neighb = range(10)
epsilons = np.linspace(0.1, 0.4, 7)
original_accuracy = np.empty((len(ns_neighb), len(epsilons)))
attacked_accuracy = np.empty((len(ns_neighb), len(epsilons)))

for i, n_neighb in enumerate(ns_neighb):
    for j, eps in enumerate(epsilons):
        m_ = DeltaEnsemble(m, n_neighb = n_neighb, eps = eps)
        m_.eval()

        outputs, outputs_ = iterate.attack(m,
            iterate.mnist_step,
            iterate.mnist_attacked_step,
            device = 'cuda',
            val_set = experiment.val_set,
            batch_size = 1000,
            epoch = 1,
            writer = writer,
            torchattack=torchattacks.PGD,
            eps=0.1,
            alpha=1/255,
            steps=1,
            random_start=False
        )

        original_acc, attacked_acc = outputs['correct'].item() / len(experiment.val_set), outputs_['correct'].item() / len(experiment.val_set)
        original_accuracy[i, j] = original_acc
        attacked_accuracy[i, j] = attacked_acc

# arr = np.array([[k1, k2, v] for (k1, k2), v in original_accuracy.items()])
# np.save('original_accuracy', arr)
img = experiment.plot_to_image(experiment.plot_trend(original_accuracy, 'original accuracy', ns_neighb, epsilons, 'num of neighbours', 'size of delta'))
writer.add_image('original accuracy', img, 1)

img = experiment.plot_to_image(experiment.plot_trend(attacked_accuracy, 'attacked accuracy', ns_neighb, epsilons, 'num of neighbours', 'size of delta'))
writer.add_image('attacked accuracy', img, 1)
# arr = np.array([[k1, k2, v] for (k1, k2), v in attacked_accuracy.items()])
# np.save('attacked_accuracy', arr)





    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')

    # print(original_accuracy)
    # ax.scatter(original_accuracy[:, 0], original_accuracy[:, 1], original_accuracy[:, 2])
    # ax.scatter(attacked_accuracy[:, 0], attacked_accuracy[:, 1], attacked_accuracy[:, 2])
    # plt.xlabel('num of neighbours')
    # plt.ylabel('size of delta')
    # ax.set_zlabel('accuracy')
    # ax.legend(['original accuracy', 'attacked accuracy'])
    # # plt.show()
    # return fig