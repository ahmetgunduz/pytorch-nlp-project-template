import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchnlp.word_to_vector import GloVe, FastText


class GloveEmbedding(nn.Module):

    """ Wrapper class for text generating RNN """

    def __init__(self,
                 vocab=None,
                 name='840B',
                 dim=300,
                 trainable=False,
                 ):

        super(GloveEmbedding, self).__init__()

        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.name = name
        self.dim = dim

        vectors = GloVe(name=self.name, dim=self.dim)
        self.weights = torch.zeros(self.vocab_size, vectors.dim)
        for idx in range(self.vocab_size):
            self.weights[idx, :] = vectors[self.vocab[idx]]

        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.embedding.weight.data = torch.Tensor(self.weights)

        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, batch):
        embeds = self.embedding(batch)
        return embeds
