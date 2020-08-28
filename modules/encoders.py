import os
import pickle

import numpy as np 
import torch 
from torch import nn


class FineTuneFasterRcnnFc7(nn.Module):
    """ Trainable module for the last fc layer of the faster R-CNN, which is to be fine-tuned. """

    def __init__(self, weights_file, bias_file):
        super(FineTuneFasterRcnnFc7, self).__init__()        
    
        if not os.path.exists(weights_file) or not os.path.exists(bias_file):
            raise FileNotFoundError

        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        
        # Reconstruct last fc7 layer using pretrained params
        # weights.shape = (out_features, in_features)
        # bias.shape = (out_features,)
        self.in_features = weights.shape[1]
        self.out_features= bias.shape[0]

        self.fc7 = nn.Linear(self.in_features, self.out_features)
        self.fc7.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float), requires_grad=True)
        self.fc7.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float), requires_grad=True)
    
    def forward(self, img_features):
        output = nn.functional.relu(self.fc7(img_features))
        return output

