from utils import experiment, iterate
from mnist_models import CNN
import torch.nn.functional as F

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

m = CNN()

iterate.train(m,
    iterate.mnist_rand_step,
    device = 'cpu',
    train_set = experiment.train_set,
    batch_size = 1000,
    optimizer = torch.optim.Adam(m.parameters(), lr = 0.001),
    epoch = 10,
    writer = writer
)

# utils.train_model(m, loss_fn, batchSize, experiment.train_set, experiment.val_set, optimizer, num_epochs)

# print('baseline: ')
# utils.attack_model(m, loss_fn, 1000, utils.valset, 400)

# import numpy as np

# original_accuracy = {}
# attacked_accuracy = {}

# for i in range(1, 10):
#     for eps in np.linspace(0.1, 0.4, 7):
#         eps = round(eps, 2)
#         m_ = DeltaEnsemble(m, n_neighb = i, eps = eps)
#         m_.eval()
#         original_acc, _, attacked_acc, _, _ = utils.attack_model(m_, loss_fn, int(40000 // max(1, i)), utils.valset, 400)
#         print(i, eps, original_acc, attacked_acc)
#         original_accuracy[i, eps] = original_acc
#         attacked_accuracy[i, eps] = attacked_acc

# arr = np.array([[k1, k2, v] for (k1, k2), v in original_accuracy.items()])
# np.save('original_accuracy', arr)

# arr = np.array([[k1, k2, v] for (k1, k2), v in attacked_accuracy.items()])
# np.save('attacked_accuracy', arr)