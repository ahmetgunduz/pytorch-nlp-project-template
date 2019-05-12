import torch.nn as nn
from abc import abstractmethod


class BaseEmbedding(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
