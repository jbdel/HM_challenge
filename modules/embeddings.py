from copy import deepcopy

import numpy as np 
import torch 
from torch import nn

from transformers.modeling_bert import BertEmbeddings

class VisualBertEmbeddings(BertEmbeddings):
    """ Construct the embeddings from textual and visual inputs for Visual Bert model.
    
    Textual inputs: input_ids, segment_ids and position_ids,
        each of shape (batch_size, max_seq_length)

    Visual inputs: img_features of shape (batch_size, img_features_number, img_features_dim)
    
    Note: There is no text-image alignement provided -> img_positions are just zeros
          Also, there is no image mask provided -> img_segments are just zeros
    Is is correct?
    Check in mmf: visual_embedding_dim, visual_embeddings_type, position_embeddings_visual, image_text_alignment

    See also https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
            https://github.com/facebookresearch/mmf/blob/67390cb41cf3275bd82c68785dbf5483c7034b41/mmf/modules/embeddings.py#L306 
    
    """
    
    def __init__(self, config):
        # Initiliaze BertEmbeddings class for textual embeddings
        super(VisualBertEmbeddings, self).__init__(config)

        # Initialize layers for visual embeddings
        self.visual_input_embeddings = nn.Linear(
                            in_features=config.visual_embedding_dim, 
                            out_features=config.hidden_size
                        )
        self.visual_segment_embeddings = nn.Embedding(
                            num_embeddings=config.type_vocab_size, 
                            embedding_dim=config.hidden_size
                        )

        self.visual_position_embeddings = nn.Embedding(
                            num_embeddings=config.max_position_embeddings, 
                            embedding_dim=config.hidden_size
                        )
    
    def special_initialize(self):
        """ Initialize visual_segment_embeddings and visual_position_embeddings layers from
            BertEmbeddings params for segment_embeddings and position_embeddings layers. """
        self.visual_segment_embeddings.weight = nn.Parameter(deepcopy(self.token_type_embeddings.weight.data), requires_grad=True)
        self.visual_position_embeddings.weight = nn.Parameter(deepcopy(self.position_embeddings.weight.data), requires_grad=True)

    def forward(self, input_ids, segment_ids, img_features):
        # First create embeddings for textual inputs
        seq_length = input_ids.shape[1]
        device = input_ids.device 
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_embeddings = self.word_embeddings(input_ids)
        segment_embeddings = self.token_type_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)

        text_embeddings = input_embeddings + segment_embeddings + position_embeddings

        # Then deal with embeddings for visual inputs
        visual_input_shape = img_features.shape[:-1]
        img_segments = torch.zeros(visual_input_shape, dtype=torch.long, device=device)
        img_positions = torch.zeros(visual_input_shape, dtype=torch.long, device=device)

        visual_input_embeddings = self.visual_input_embeddings(img_features)
        visual_segment_embeddings = self.visual_segment_embeddings(img_segments)
        visual_position_embeddings = self.visual_segment_embeddings(img_positions)

        visual_embeddings = visual_input_embeddings + visual_segment_embeddings + visual_position_embeddings
        
        # Concatenate the text and visual embeddings one after each other
        embeddings = torch.cat((text_embeddings, visual_embeddings), dim=1)

        # Pass through BertLayerNorm and Dropout layers as in BertEmbeddings class
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


