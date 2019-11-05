import pdb

import torch
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    target = target.view(-1)
    return F.cross_entropy(output, target)


birary_cross_entropy_with_logits = BCEWithLogitsLoss()
