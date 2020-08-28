import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils.utils_loader import data_transform_rn152
from PIL import Image


class HMResnet(Dataset):
    """ Hateful Memes Dataset for Resnet."""

    def __init__(self, name, args, data_path="data"):

        assert name in ["train", "dev", "test"]

        self.args = args
        self.name = name

        self.file_path = os.path.join(data_path, name + "_data")
        self.df = pd.read_pickle(self.file_path)

        # Transform np.array img to PIL img and apply transform for rn152
        self.df['img'] = self.df['img'].map(lambda img: data_transform_rn152(Image.fromarray(img)))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}
        sample['img'] = self.df.iloc[idx, 1]
        sample['text'] = self.df.iloc[idx, -1]

        if self.name == "test":
            sample['label'] = np.array([])
        else:
            sample['label'] = torch.tensor([self.df.iloc[idx, 2]])

        return sample

    def getIdNumber(self, idx):
        """ Returns the Id number of the image corresponding to index """
        return self.df.iloc[idx, 0]