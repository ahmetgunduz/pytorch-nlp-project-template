import os
import pdb
import sys
import random
import torch
from torch import utils
from torch.utils.data import Dataset
from utils.vocab import Vocabulary



class RickAndMortyDataset(Dataset):

    """ Wrapper class to process and produce training samples """

    def __init__(self, data_dir, seq_length, vocab_size=None, vocab=None, training=False):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.vocab = Vocabulary()
        with open(os.path.join(data_dir, 'rick_and_morty.txt'), 'r', encoding="utf-8") as f:
            self.text = f.read()

        if vocab is not None:
            if isinstance(vocab, str):
                self.vocab.load(vocab)
            elif isinstance(vocab, Vocabulary):
                self.vocab = vocab
        elif os.path.exists(os.path.join(data_dir, "vocab.pkl")):
            self.vocab.load(os.path.join(data_dir, "vocab.pkl"))
        else:
            self.vocab.add_text(self.text)
            self.vocab.save(os.path.join(data_dir, "vocab.pkl"))

        if vocab_size is not None:
            self.vocab = self.vocab.most_common(vocab_size-2)

        self.text = self.vocab.clean_text(self.text)
        self.tokens = self.vocab.tokenize(self.text)

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = [self.vocab[word]
             for word in self.tokens[idx:idx + self.seq_length]]
        y = [self.vocab[self.tokens[idx + self.seq_length]]]
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x, y
