import os
from glob import glob

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.vocab import Vocabulary


class BaseDataset(Dataset):
    """ Wrapper class to process and produce training samples """

    def validation(self):
        """
        Set validation dataset

        :return: Boolean
        """
        raise NotImplementedError

    def train(self):
        """
         Set train dataset

         :return: Boolean
         """
        raise NotImplementedError

    def test(self):
        """
         Set test dataset

         :return: Boolean
         """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
         Get item

         :return: Dataset outputs
        """
        raise NotImplementedError