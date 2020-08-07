import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class HMVisualBERTDataset(Dataset):
    """ Hateful Memes Dataset for Visual BERT.

        Data structure: pandas Dataframe with columns:
            - "id": id number of the data sample - format: numpy.int64 
            - "img_name": path for image (unused) - format: str
            - "features": img features - format: numpy.array of shape (100, 2048) and type np.float32
            - "text": text - format: str

        Check img_features_extractor.ipynb to see how the image features are extracted
    """

    def __init__(self, name, args, data_path='data'):

        assert name in ['train', 'dev', 'test']

        self.args = args
        self.name = name

        self.file_path = os.path.join(data_path, name + '_data')
        self.df = pd.read_pickle(self.file_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}

        # torch.tensor automatically creates a copie of the data
        sample['features'] = torch.tensor(self.df['features'][idx], dtype=torch.float)

        # TODO: add text processing with BERT
        sample['text'] = self.df['text'][idx]


        if self.name == 'test':
            sample['label'] = torch.tensor(np.array([]), dtype=torch.float)
        else:
            sample['label'] = torch.tensor(self.df['label'][idx], dtype=torch.float)

        return sample

    def getIdNumber(self, idx):
        """ Returns the id number of the image corresponding to index """
        return torch.tensor(self.df['id'][idx], dtype=torch.int)
