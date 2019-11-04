import os

import torch
from torch.utils.data import Dataset

from utils.vocab import Vocabulary


class EmotionDataset(Dataset):
    """ Wrapper class to process and produce training samples """

    def __init__(
        self, data_dir, vocab_size=None, vocab=None, seq_length=40, training=False
    ):
        self.data_dir = data_dir
        self.vocab = Vocabulary()
        self.seq_length = seq_length

        with open(
            os.path.join(data_dir, "sentiment_140.csv"), "r", encoding="utf-8"
        ) as f:
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
            self.vocab = self.vocab.most_common(vocab_size - 2)

        self.text = self.vocab.clean_text(self.text)
        self.tokens = self.vocab.tokenize(self.text)
        self.new_line_indices = [-1] + [
            i for i, x in enumerate(self.tokens) if x == "<new_line>"
        ]

    def __len__(self):
        return len(self.new_line_indices) - 1

    def __getitem__(self, idx):
        x = [
            _
            for _ in self.tokens[
                self.new_line_indices[idx] + 2 : self.new_line_indices[idx + 1]
            ]
        ]
        if len(x) > self.seq_length:
            u = x[: self.seq_length]
        else:
            u = x + (["<pad>"] * (self.seq_length - len(x)))
        x = [self.vocab[_] for _ in u]
        y = [0, 0]
        y[int(self.tokens[self.new_line_indices[idx] + 1].split("__label__")[1])] = 1
        x = torch.LongTensor(x)
        y = torch.FloatTensor(y)
        return x, y
