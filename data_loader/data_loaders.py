from base import BaseDataLoader
from datasets.rickandmorty import RickAndMortyDataset
from datasets.simpsons import SimpsonsDataset
from datasets.emotion_twitter import EmotionDataset


class RickAndMortyDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        seq_length,
        batch_size,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.dataset = RickAndMortyDataset(
            data_dir=data_dir, seq_length=seq_length, vocab_size=vocab_size, vocab=vocab
        )
        super(RickAndMortyDataLoader, self).__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class EmotionDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        seq_length,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.dataset = EmotionDataset(
            data_dir=data_dir, vocab_size=vocab_size, vocab=vocab, seq_length=seq_length
        )
        super(EmotionDataLoader, self).__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class SimpsonsDataLOader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        seq_length,
        batch_size,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.dataset = SimpsonsDataset(
            data_dir=data_dir, seq_length=seq_length, vocab_size=vocab_size, vocab=vocab
        )
        super(SimpsonsDataLOader, self).__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
