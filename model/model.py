import torch.nn as nn

from base import BaseModel


class MortyFire(BaseModel):
    """ Wrapper class for text generating RNN """

    def __init__(
        self,
        lstm_size,
        seq_length,
        num_layers,
        vocab=None,
        lstm_dropout=0.3,
        fc_dropout=0.2,
        bidirectional=False,
        embedding=None,
    ):

        super(MortyFire, self).__init__()
        #         pdb.set_trace()
        assert vocab is not None, "Please specify vocab"
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, 200)
            self.embed_size = 200
            self.vocab_size = embedding.num_embeddings

        else:
            self.embedding = embedding
            self.embed_size = self.embedding.dim
            self.vocab_size = len(embedding.vocab)

        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.seq_length = seq_length

        self.bidirectional = bidirectional

        # if embeddings is not None:
        #     self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings))

        #     self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            self.embed_size,
            lstm_size,
            num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_size * 2, int(self.vocab_size / 2)),
            nn.Dropout(fc_dropout),
            nn.Linear(int(self.vocab_size / 2), self.vocab_size),
        )

    def forward(self, batch):
        batch_size = batch.size(0)
        embeds = self.embedding(batch)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_size * 2)
        drop = self.lstm_dropout(lstm_out)
        output = self.classifier(drop)
        output = output.view(batch_size, -1, self.vocab_size)
        out = output[:, -1]
        return out


class EmotionFire(BaseModel):
    """ Wrapper class for text generating RNN """

    def __init__(
        self,
        lstm_size,
        seq_length,
        num_layers,
        out_dim=2,
        vocab=None,
        lstm_dropout=0.3,
        fc_dropout=0.2,
        bidirectional=False,
        embedding=None,
    ):

        super(EmotionFire, self).__init__()
        #         pdb.set_trace()
        assert vocab is not None, "Please specify vocab"
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, 200)
            self.embed_size = 200
            self.vocab_size = embedding.num_embeddings

        else:
            self.embedding = embedding
            self.embed_size = self.embedding.dim
            self.vocab_size = len(embedding.vocab)

        self.out_dim = out_dim
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.seq_length = seq_length

        self.bidirectional = bidirectional

        # if embeddings is not None:
        #     self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings))

        #     self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            self.embed_size,
            lstm_size,
            num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_size * 2, int(self.vocab_size / 2)),
            nn.Dropout(fc_dropout),
            nn.Linear(int(self.vocab_size / 2), self.out_dim),
        )

    def forward(self, batch):
        batch_size = batch.size(0)
        embeds = self.embedding(batch)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_size * 2)
        drop = self.lstm_dropout(lstm_out)
        output = self.classifier(drop)
        output = output.view(batch_size, -1, self.out_dim)
        out = output[:, -1]
        return out
