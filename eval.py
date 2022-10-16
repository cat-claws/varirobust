import sys
sys.path.append('.')
import utils
from utils import attack_model, get_model_name, Ensem

import torch
import os

checkpoints = os.listdir('checkpoints')

single_robustness = []
for checkpoint in checkpoints:
    m = torch.load('checkpoints/' + checkpoint)
    mname = get_model_name(m)
    batchSize = 100
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 40

    print(mname)
    single_robustness.append(attack_model(m, loss_fn, batchSize, utils.valset, num_epochs))

double_robustness = []
for i, c1 in enumerate(checkpoints):
    for j, c2 in enumerate(checkpoints):
        m1 = torch.load('checkpoints/' + c1)
        m2 = torch.load('checkpoints/' + c2)
        m = Ensem(m1, m2)
        mname = get_model_name(m)
        batchSize = 100
        loss_fn = torch.nn.CrossEntropyLoss()
        num_epochs = 40

        double_robustness.append([single_robustness[i], single_robustness[j], attack_model(m, loss_fn, batchSize, utils.valset, num_epochs)])