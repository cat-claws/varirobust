import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import torchattacks
import numpy as np

get_model_name = lambda m: str(m.__class__).split('.')[-1][:-2]

class MNISTFast(datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.data_ = []
        for index in range(len(self.data)):
            img, target = super().__getitem__(index)
            self.data_.append((img.clone().detach(), target))
            
    def __getitem__(self, index: int):
        return self.data_[index]


# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
    ])

trainset = MNISTFast('Dataset', train=True, download=True, transform=transform)
valset = MNISTFast('Dataset', train=False, transform=transform)
# trainset = datasets.MNIST('Dataset', train=True, download=True, transform=transform)
# valset = datasets.MNIST('Dataset', train=False, transform=transform)

def uniform_perturb(x, eps = 0.1):
    grad = torch.rand_like(x)
    ub = torch.clamp(x + eps, min = 0, max = 1)
    lb = torch.clamp(x - eps, min = 0, max = 1)
    delta = ub - lb
    return delta * grad + lb  

def train_model(model, loss_fn, batchSize, trainset, valset, optimizer, num_epochs):
  

    # Shuffling is needed in case dataset is not shuffled by default.
    train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                                batch_size = batchSize,
                                                shuffle = True)
    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                                batch_size = batchSize,
                                                shuffle = False) # No need.

    # Define number of epochs.
    N = num_epochs

    # log accuracies and losses.
    train_accuracies = []; val_accuracies = []
    train_losses = []; val_losses = []

    # GPU enabling.
    model = model.cuda()
    # loss_fn = loss_fn.cuda()


    # Training loop. Please make sure you understand every single line of code below.
    # Go back to some of the previous steps in this lab if necessary.
    for epoch in range(0, N):
        correct = 0.0
        cum_loss = 0.0

        # Make a pass over the training data.
        model.train()
        for (i, (inputs, labels)) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass. (Prediction stage)
            scores = model(inputs + 0.2 * torch.randn_like(inputs))
            loss = loss_fn(scores, labels)

            # Count how many correct in this batch.
            max_scores, max_labels = scores.max(1)
            correct += (max_labels == labels).sum().item()
            cum_loss += loss.item()

            # Zero the gradients in the network.
            optimizer.zero_grad()

            #Backward pass. (Gradient computation stage)
            loss.backward()

            # Parameter updates (SGD step) -- if done with torch.optim!
            optimizer.step()

            # Parameter updates (SGD step) -- if done manually!
            # for param in model.parameters():
            #   param.data.add_(-learningRate, param.grad)

            # Logging the current results on training.
            if (i + 1) % 100 == 0:
                print('Train-epoch %d. Iteration %05d / %05d, Avg-Loss: %.4f, Accuracy: %.4f' % 
                    (epoch, i + 1, len(train_loader), cum_loss / (i + 1), correct / ((i + 1) * batchSize)))

        train_accuracies.append(correct / len(trainset))
        train_losses.append(cum_loss / (i + 1))   

        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        model.eval()
        for (i, (inputs, labels)) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass. (Prediction stage)
            scores = model(inputs)
            cum_loss += loss_fn(scores, labels).item()

            # Count how many correct in this batch.
            max_scores, max_labels = scores.max(1)
            correct += (max_labels == labels).sum().item()

        val_accuracies.append(correct / len(valset))
        val_losses.append(cum_loss / (i + 1))

        # Logging the current results on validation.
        print('Validation-epoch %d. Avg-Loss: %.4f, Accuracy: %.4f' % 
            (epoch, cum_loss / (i + 1), correct / len(valset)))
        
        

def attack_model(model, loss_fn, batchSize, valset, num_epochs):
  

    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset = valset,
                                                batch_size = batchSize,
                                                shuffle = False, num_workers=2) # No need.

    # GPU enabling.
    model = model.cuda()
    # loss_fn = loss_fn.cuda()



    # Training loop. Please make sure you understand every single line of code below.
    # Go back to some of the previous steps in this lab if necessary.
    original_correct = 0.0
    original_cum_loss = 0.0

    attacked_correct = 0.0
    attacked_cum_loss = 0.0

    # Make a pass over the training data.
    model.eval()
    for (i, (inputs, labels)) in enumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward pass. (Prediction stage)
        scores = model(inputs)
        loss = loss_fn(scores, labels)

        # Count how many correct in this batch.
        max_scores, max_labels = scores.max(1)
        original_correct += (max_labels == labels).sum().item()
        original_cum_loss += loss.item()
        
        
        # Attack the model using PGD
        atk = torchattacks.PGD(model, eps=0.1, alpha=1/255, steps=num_epochs, random_start=False)
        adv_images = atk(inputs, labels)

        # Forward pass. (Prediction stage)
        scores = model(adv_images)
        loss = loss_fn(scores, labels)

        # Count how many correct in this batch.
        max_scores, max_labels = scores.max(1)
        attacked_correct += (max_labels == labels).sum().item()
        attacked_cum_loss += loss.item()

        original_accuracy = original_correct / len(valset)
        original_losses = original_cum_loss / (i + 1)

        attacked_accuracy = attacked_correct / len(valset)
        attacked_losses = attacked_cum_loss / (i + 1)      

    print('Before. Avg-Loss: %.4f, Accuracy: %.4f' % 
            (original_cum_loss / (i + 1), original_correct / len(valset)))
    print('After. Avg-Loss: %.4f, Accuracy: %.4f' % 
            (attacked_cum_loss / (i + 1), attacked_correct / len(valset)))
    
    return original_accuracy, original_losses, attacked_accuracy, attacked_losses, adv_images

class Ensem(torch.nn.Module):

    def __init__(self, m1, m2):
        super(Ensem, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, data):
        return (self.m1(data).softmax(-1) + self.m2(data).softmax(-1))/2

def get_heat_map(x, model):
    x1 = x.clone().detach()
    x1.requires_grad = True
    model.eval()
    d1 = model(x1)
    d2 = torch.rand_like(d1).uniform_(-100, 100).softmax(-1)
    loss = F.kl_div(d1.log(), d2)
    loss.backward()
    return x1.grad

class DeltaEnsemble(torch.nn.Module):
    def __init__(self, m, eps = 0.1, n_neighb = 0):
        super(DeltaEnsemble, self).__init__()
        self.m = m
        self.eps = eps
        self.n_neighb = n_neighb

    def _get_neighb(self, x, n_neighb):
        all_inputs = [x]
        for k in range(n_neighb):
            grad = torch.sigmoid(torch.rand_like(x).uniform_(-200, 200))
            ub = torch.clamp(x + self.eps, min = 0, max = 1)
            lb = torch.clamp(x - self.eps, min = 0, max = 1)
            delta = ub - lb
            x2 = delta * grad + lb
            all_inputs.append(x2)
        return torch.stack(all_inputs)

    def _cam(self, x):
        x_ = x.clone().detach()
        x_.requires_grad = True
        z_ = self.m._feature(x_)
        z = z_.detach()
        z = z +  torch.randn_like(z).normal_(std = z.std() / 10)
        loss_z = F.mse_loss(z_, z)
        loss_z.backward()
        return x_.grad 

    def _get_neighb_with_grad(self, x, n_neighb):
        cam_abs = self._cam(x).abs()
        cam_mask = cam_abs > np.percentile(cam_abs.cpu(), 75)

        x = x.unsqueeze(0)
        x_ = x.repeat(n_neighb, 1, 1, 1, 1)
        x_ = x + torch.randn_like(x_).sign() * self.eps * cam_mask
        x_ = torch.clamp(x_, min = 0, max = 1)
        x_ = torch.cat((x, x_), dim = 0)
        return x_

    def _get_neighb_uniform(self, x, n_neighb):
        x = x.unsqueeze(0)
        x_ = x.repeat(n_neighb, 1, 1, 1, 1)
        ub = torch.clamp(x + self.eps, min = 0, max = 1)
        lb = torch.clamp(x - self.eps, min = 0, max = 1)
        x_ = (ub - lb) * torch.rand_like(x_) + lb
        x_ = torch.cat((x, x_), dim = 0)
        return x_

    def _predict_neighb(self, x, n_neighb):
        all_inputs = self._get_neighb(x, n_neighb)
        if (n_neighb + 1) * len(x) <= 10000:
            outputs = self.m(all_inputs.view((n_neighb + 1) * len(x), *x.shape[1:])).view((n_neighb + 1), len(x), -1)
        else:
            outputs = torch.stack([self.m(i) for i in all_inputs])
        return outputs

    def forward(self, x, n_neighb = -1):
        if n_neighb == -1:
            n_neighb = self.n_neighb

        if n_neighb == 0:
            return self.m(x)
        else:            
            outputs = self._predict_neighb(x, n_neighb)
            return sum(outputs) / n_neighb