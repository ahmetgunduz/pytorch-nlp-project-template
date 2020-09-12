import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    target = target.view(-1)
    return F.cross_entropy(output, target)


def none_loss(loss):
    return loss


bce_loss = BCELoss()
mse_loss = MSELoss()
binary_cross_entropy_with_logits = BCEWithLogitsLoss()