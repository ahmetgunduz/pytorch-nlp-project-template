from copy import deepcopy

from base import BaseTestDataLoader, BaseTrainDataLoader
from datasets.email_spam import EmailSpamDataset
from datasets.rickandmorty import RickAndMortyDataset


class RickAndMortyTrainDataLoader(BaseTrainDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        seq_length,
        vocab_from_pretrained=False,
        do_lower_case=False,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.dataset = RickAndMortyDataset(
            data_dir=data_dir,
            vocab_size=vocab_size,
            vocab=vocab,
            seq_length=seq_length,
            vocab_from_pretrained=vocab_from_pretrained,
            do_lower_case=do_lower_case,
            training=training,
        )

        self.dl_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        super(RickAndMortyTrainDataLoader, self).__init__(
            self.dataset, validation_split=validation_split, **self.dl_kwargs
        )

    def get_validation(self):
        dataset = deepcopy(self.dataset)
        isSet = dataset.validation()
        if isSet:
            return BaseTestDataLoader(self.dataset, **self.dl_kwargs)
        else:
            return self.split_validation()


class RickAndMortyTestDataLoader(BaseTestDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        seq_length,
        vocab_from_pretrained=False,
        do_lower_case=False,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=False,
    ):
        self.dataset = RickAndMortyDataset(
            data_dir=data_dir,
            vocab_size=vocab_size,
            vocab=vocab,
            seq_length=seq_length,
            vocab_from_pretrained=vocab_from_pretrained,
            do_lower_case=do_lower_case,
            training=training,
        )

        self.dl_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }

        super(RickAndMortyTestDataLoader, self).__init__(
            self.dataset,  **self.dl_kwargs
        )

class EmailSpamTrainDataLoader(BaseTrainDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        seq_length,
        vocab_from_pretrained=False,
        do_lower_case=False,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.dataset = EmailSpamDataset(
            data_dir=data_dir,
            vocab_size=vocab_size,
            vocab=vocab,
            seq_length=seq_length,
            vocab_from_pretrained=vocab_from_pretrained,
            do_lower_case=do_lower_case,
            training=training,
        )

        self.dl_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        super(EmailSpamTrainDataLoader, self).__init__(
            self.dataset, validation_split=validation_split, **self.dl_kwargs
        )

    def get_validation(self):
        dataset = deepcopy(self.dataset)
        isSet = dataset.validation()
        if isSet:
            return BaseTestDataLoader(self.dataset, **self.dl_kwargs)
        else:
            return self.split_validation()


class EmailSpamTestDataLoader(BaseTestDataLoader):
    """
    data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        seq_length,
        vocab_from_pretrained=False,
        do_lower_case=False,
        vocab_size=None,
        vocab=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=False,
    ):
        self.dataset = EmailSpamDataset(
            data_dir=data_dir,
            vocab_size=vocab_size,
            vocab=vocab,
            seq_length=seq_length,
            vocab_from_pretrained=vocab_from_pretrained,
            do_lower_case=do_lower_case,
            training=training,
        )

        self.dl_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }

        super(EmailSpamTestDataLoader, self).__init__(
            self.dataset,  **self.dl_kwargs
        )
