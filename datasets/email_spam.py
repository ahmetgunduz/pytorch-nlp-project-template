import os
from glob import glob

import torch
import pandas as pd

from base import BaseDataset
from utils.vocab import Vocabulary


class EmailSpamDataset(BaseDataset):
    """ Wrapper class to process and produce training samples """

    def __init__(
        self,
        data_dir,
        vocab_size=None,
        vocab=None,
        seq_length=40,
        training=False,
        vocab_from_pretrained="bert-base-uncased",
        do_lower_case=True,
    ):

        self.data_dir = data_dir
        self.vocab = Vocabulary(vocab_from_pretrained, do_lower_case)
        self.seq_length = seq_length

        data_all = pd.read_csv(os.path.join(self.data_dir, "combined-data.csv"), sep=' ', header=None, encoding="cp1252")
        data_all[1] = data_all[1] + " " + data_all[2]
        data_all = data_all[[0, 1]]
        data_all.columns = ['label', 'text']
        data_all = data_all[['text', 'label']]
        data_all = data_all[~data_all.text.isna()]
        data_all.label = data_all.label.apply(lambda x: int(x[-1]))
        data_all.text = data_all.text.apply(lambda x: x.lower())

        data_all = data_all.sample(1000)
        
        self.train_df = data_all.copy() #pd.DataFrame({"text": [], "label": []})
        self.val_df = pd.DataFrame({"text": [], "label": []})
        self.test_df = data_all.copy() # pd.DataFrame({"text": [], "label": []}) #data_all.copy()

        del data_all

        if training:
            self.train()
            if vocab is not None:
                if isinstance(vocab, str):
                    self.vocab.load(vocab)
                elif isinstance(vocab, Vocabulary):
                    self.vocab = vocab
            elif os.path.exists(os.path.join(data_dir, "vocab.pkl")):
                self.vocab.load(os.path.join(data_dir, "vocab.pkl"))
            else:
                self.vocab.add_text(
                    " ".join(pd.concat([self.train_df, self.val_df], sort=False).text.values)
                )
                self.vocab.save(os.path.join(data_dir, "vocab.pkl"))
        else:
            self.test()
            if vocab is not None:
                if isinstance(vocab, str):
                    self.vocab.load(vocab)
                elif isinstance(vocab, Vocabulary):
                    self.vocab = vocab
            elif os.path.exists(os.path.join(data_dir, "vocab.pkl")):
                self.vocab.load(os.path.join(data_dir, "vocab.pkl"))
            else:
                raise(Exception("Vocab file is not specified in test mode!"))
        
        if vocab_size is not None:
                self.vocab = self.vocab.most_common(vocab_size - 2)

    def validation(self):
        self.text = self.val_df.text.values
        self.labels = self.val_df.label.values
        self.len = len(self.val_df)
        return True

    def train(self):
        self.text = self.train_df.text.values
        self.labels = self.train_df.label.values
        self.len = len(self.train_df)
        return True

    def test(self):
        self.text = self.test_df.text.values
        self.labels = self.test_df.label.values
        self.len = len(self.test_df)
        return True

    def __len__(self):
        return self.len - 1 if self.len else 0

    def __getitem__(self, idx):
        y = self.labels[idx]
        text = self.text[idx]

        text = self.vocab.clean_text(text)
        input_ids, attention_mask, segment_ids = self.format_in_text(text)
        y = torch.LongTensor([y])

        return input_ids, attention_mask, segment_ids, y
    
    def format_in_text(self, text):
        text = self.vocab.clean_text(text)
        tokens_a = self.vocab.tokenize(text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.seq_length - 2:
            tokens_a = tokens_a[: (self.seq_length - 2)]

        tokens = (
                [self.vocab.tokenizer.cls_token]
                + tokens_a
                + [self.vocab.tokenizer.sep_token]
        )
        segment_ids = [0] * len(tokens)
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.vocab[x] for x in tokens]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [self.vocab.tokenizer.pad_token_id] * (
                self.seq_length - len(input_ids)
        )
        input_ids += padding
        attention_mask += padding
        segment_ids += padding


        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        segment_ids = torch.LongTensor(segment_ids)
        return input_ids, attention_mask, segment_ids
