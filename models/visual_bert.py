import os
from copy import deepcopy

import torch 
from torch import nn

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
    BertPooler
)
from modules.embeddings import VisualBertEmbeddings

class VisualBertModel(BertPreTrainedModel):
    def __init__(self,
        config,
        visual_embedding_dim=2048,
    ):

        super(VisualBertModel, self).__init__(config)
        self.config=config
        config.visual_enbedding_dim = visual_embedding_dim
        self.embeddings = VisualBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

