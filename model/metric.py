import numpy as np
import torch
from sklearn.metrics import accuracy_score

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """
    # pdb.set_trace()
    _, predicted = torch.max(outputs, 1)
    if len(labels.shape) > 1:
        _, labels = torch.max(labels, 1)
    total = len(labels)
    correct = (predicted == labels).sum()
    accuracy = float(correct) / total
    return accuracy

def bce_loss(outputs, labels):

    if outputs.is_cuda:
        outputs = outputs.cpu().detach()
    if labels.is_cuda:
        labels = labels.cpu()

    if outputs.shape[0] != 1:
        predicted = np.argmax(outputs, axis=1)
    else:
        predicted = outputs
    accuracy = accuracy_score(y_true=labels, y_pred=predicted)
    return accuracy


def mse(outputs, labels):
    return torch.mean((outputs - labels) ** 2)


def mae(outputs, labels):
    return torch.mean(torch.abs(outputs - labels))