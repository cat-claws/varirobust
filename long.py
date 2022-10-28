import utils
from utils import DeltaEnsemble
from mnist_models import TwoLayerNN, MLP, CNN, MLPBN, ConvNet, LeNet, LeNet5
import torch.nn.functional as F

import torch

m = CNN()
batchSize = 1000
loss_fn = lambda x, y: F.nll_loss(torch.log(x + 1e-9), y)
learningRate = 0.001

optimizer = torch.optim.Adam(m.parameters(), lr = learningRate)

num_epochs = 1000

utils.train_model(m, loss_fn, batchSize, utils.trainset, utils.valset, optimizer, num_epochs)

print('baseline: ')
utils.attack_model(m, loss_fn, 1000, utils.valset, 400)

import numpy as np

original_accuracy = {}
attacked_accuracy = {}

for i in range(1, 10):
    for eps in np.linspace(0.1, 0.4, 7):
        eps = round(eps, 2)
        m_ = DeltaEnsemble(m, n_neighb = i, eps = eps)
        m_.eval()
        original_acc, _, attacked_acc, _, _ = utils.attack_model(m_, loss_fn, int(40000 // max(1, i)), utils.valset, 400)
        print(i, eps, original_acc, attacked_acc)
        original_accuracy[i, eps] = original_acc
        attacked_accuracy[i, eps] = attacked_acc

arr = np.array([[k1, k2, v] for (k1, k2), v in original_accuracy.items()])
np.save('original_accuracy', arr)

arr = np.array([[k1, k2, v] for (k1, k2), v in attacked_accuracy.items()])
np.save('attacked_accuracy', arr)