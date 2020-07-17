from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from utils.utils_loader import data_transform_rn152, img_name_to_PIL_img


class HMDataset(Dataset):
    """ Hateful Memes Dataset."""

    def __init__(self, name, args, data_path="HMDataset/data"):

        assert name in ["train", "dev", "test"]

        self.args = args
        self.name = name

        self.file_path = os.path.join(data_path, name + ".jsonl")
        self.df = pd.read_json(self.file_path, lines=True)

        # TODO changer ici en un gros np.load(), appliquer transform une fois pour toute sur
        #  chaque entrée de l'array
        self.df['img'] = self.df['img'].map(lambda img_name:
                                            img_name_to_PIL_img(img_name,
                                                                data_path,
                                                                data_transform_rn152))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}
        sample['img'] = self.df.iloc[idx, 1]
        sample['text'] = self.df.iloc[idx, -1]

        if self.name == "test":
            sample['label'] = np.array([])
        else:
            sample['label'] = torch.FloatTensor([self.df.iloc[idx, 2]])

        return sample

    def getIdNumber(self, idx):
        """ Returns the Id number of the image corresponding to index """
        return self.df.iloc[idx, 0]