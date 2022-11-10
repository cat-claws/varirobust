import torch

def augmented_accuracy(net, inputs, labels, eps, n_samples, beta):

    batch_correct_ls = []
    for _ in range(n_samples):

        # sample deltas and pass perturbed images through model
        output = net(torch.clamp(inputs + 2 * eps * torch.rand_like(inputs) - eps, 0.0, 1.0))
        pert_pred = output.argmax(dim=1, keepdim=True)

        # unreduced predictions
        pert_correct = pert_pred.eq(labels.view_as(pert_pred))

        batch_correct_ls.append(pert_correct)

    batch_correct = torch.sum(torch.hstack(batch_correct_ls), dim=1)

    aug_acc = pert_correct.sum()

    # Calculate the quantile accuracy for the augmented samples.
    beta_quant_indiv_accs = torch.where(
        batch_correct > (1 - beta) * 100,
        torch.ones_like(batch_correct),
        torch.zeros_like(batch_correct))

    beta_quant_acc = beta_quant_indiv_accs.float().sum()

    return aug_acc, beta_quant_acc