import torch.nn.functional as F
from torch.nn import Sigmoid, CrossEntropyLoss, BCEWithLogitsLoss, BCELoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


cross_entropy = CrossEntropyLoss()

birary_cross_entropy_with_logits = BCEWithLogitsLoss()


def nll_loss(output, target):
    return F.nll_loss(output, target)
