import sys
sys.path.append('.')

import utils
from utils import train_model, get_model_name

from mnist_models import TwoLayerNN, MLP, CNN, MLPBN, ConvNet, LeNet, LeNet5

import torch

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

i = 0
for m in [CNN(), MLPBN()]:#, LeNet(), MLP(), TwoLayerNN(), ConvNet()]:
    mname = get_model_name(m)
    print(mname)

    batchSize = 100
    loss_fn = torch.nn.CrossEntropyLoss()
    learningRate = 0.001

    optimizer = torch.optim.Adam(m.parameters(), lr = learningRate)

    for k in range(15):

        if k % 5 == 0:
            m.apply(weight_reset)

        num_epochs = 3

        train_model(m, loss_fn, batchSize, utils.trainset, utils.valset, optimizer, num_epochs)

        torch.save(m, f'checkpoints/m{i:02}')

        i += 1