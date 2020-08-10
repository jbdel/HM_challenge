import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class HMVisualBertDataset(Dataset):
    """ Hateful Memes Dataset for Visual BERT.

        Data structure: pandas Dataframe with columns:
            - "id": id number of the data sample - format: numpy.int64 
            - "img_name": image path (unused) - format: str
            - "label": sample label (if train or dev) - format: 
            - "text": text - format: str
            - "img_features": image features - format: numpy.array of shape (100, 2048) and type np.float32
            - "text_encoding": text encoding - format: dic with keys/values:
                                                    "input_tokens": list[str]
                                                    "input_ids": np.array of shape (max_seq_length,)
                                                    "segment_ids": np.array of shape (max_seq_length,)
                                                    "input_mask: np.array of shape (max_seq_length,)

        Check input_preprocessing.ipynb to see how the image features and text encodings are obtained
    """

    def __init__(self, name, args):

        assert name in ['train', 'dev', 'test']

        self.args = args
        self.name = name

        self.file_path = os.path.join(self.args.datapath, name + '_data')
        self.df = pd.read_pickle(self.file_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}

        # torch.tensor automatically creates a copie of the data
        sample['img_features'] = torch.tensor(self.df['img_features'][idx], dtype=torch.float)

        sample['text_encoding'] = {}
        sample['text_encoding']['input_tokens'] = self.df['text_encoding'][idx]['input_token']
        sample['text_encoding']['input_ids'] = torch.tensor(self.df['text_encoding'][idx]['input_ids'], dtype=torch.int64)
        sample['text_encoding']['segment_ids'] = torch.tensor(self.df['text_encoding'][idx]['segment_ids'], dtype=torch.int8)
        sample['text_encoding']['input_mask'] = torch.tensor(self.df['text_encoding'][idx]['input_mask'], dtype=torch.int8)

        if self.name == 'test':
            sample['label'] = torch.tensor(np.array([]), dtype=torch.float)
        else:
            sample['label'] = torch.tensor(self.df['label'][idx], dtype=torch.float)

        return sample

    def getIdNumber(self, idx):
        """ Returns the id number of the image corresponding to index """
        return torch.tensor(self.df['id'][idx], dtype=torch.int)
