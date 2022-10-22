import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import torchattacks

get_model_name = lambda m: str(m.__class__).split('.')[-1][:-2]

# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
    ])

trainset = datasets.MNIST('Dataset', train=True, download=True, transform=transform)
valset = datasets.MNIST('Dataset', train=False, transform=transform)

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
            scores = model(inputs)
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
                                                shuffle = False) # No need.

    # Define number of epochs.
    N = num_epochs

    # log accuracies and losses.
    original_accuracies = []; attacked_accuracies = []
    original_losses = []; attacked_losses = []

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


        # Logging the current results on attacks.
        if (i + 1) % 100 == 0:
            print('Before: Partition %05d / %05d, Avg-Loss: %.4f, Accuracy: %.4f' % 
                (i + 1, len(val_loader), original_cum_loss / (i + 1), original_correct / ((i + 1) * batchSize)))
            print('After: Partition %05d / %05d, Avg-Loss: %.4f, Accuracy: %.4f' % 
                (i + 1, len(val_loader), attacked_cum_loss / (i + 1), attacked_correct / ((i + 1) * batchSize)))

        original_accuracy = original_correct / len(valset)
        original_losses = original_cum_loss / (i + 1)

        attacked_accuracy = attacked_correct / len(valset)
        attacked_losses = attacked_cum_loss / (i + 1)      
    
    return original_accuracy, original_losses, attacked_accuracy, attacked_losses, adv_images

class Ensem(torch.nn.Module):

    def __init__(self, m1, m2):
        super(Ensem, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, data):
        return (self.m1(data).softmax(-1) + self.m2(data).softmax(-1))/2

class DeltaEnsemble(torch.nn.Module):
    def __init__(self, m, eps = 0.1, n_neighb = 0):
        super(DeltaEnsemble, self).__init__()
        self.m = m
        self.eps = eps
        self.n_neighb = n_neighb

    def forward(self, x, n_neighb = -1):
        if n_neighb == -1:
            n_neighb = self.n_neighb

        if n_neighb == 0:
            return self.m(x)
        else:
            outputs = []
            for k in range(n_neighb):
                grad = torch.randn_like(x)
                adv_images = x + grad #self.eps * grad.sign()
                adv_images = torch.clamp(adv_images, min=0, max=1)
                outputs.append(self.m(adv_images))
            return sum(outputs) / n_neighb