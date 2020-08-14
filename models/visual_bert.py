import os
from copy import deepcopy

import torch 
from torch import nn

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform
)
from modules.embeddings import VisualBertEmbeddings

from modules.encoders import FineTuneFasterRcnnFc7

class VisualBertModel(BertPreTrainedModel):
    """ Explanation."""

    def __init__(
        self,
        config,
        visual_embedding_dim=2048
    ):
        super(VisualBertModel, self).__init__(config)
        config.visual_embedding_dim = visual_embedding_dim
        self.config = config
        self.embeddings = VisualBertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        # Mask to nullify selected heads of the self-attention modules
        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]

        # Special initialize for embeddings
        # self.embeddings.special_initialize()

        # Initialize the weights
        # self.init_weights()
    
    def forward(
        self,
        input_ids,
        segment_ids,
        img_features,
        input_mask
    ):  

        # Add image mask to input mask to get attention mask for the whole input (CHECK)
        img_mask = torch.ones(img_features.shape[:-1]).long()

        attention_mask = torch.cat((input_mask, img_mask), dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Textual and visual input embedding
        embedding_output = self.embeddings(
            input_ids=input_ids,
            segment_ids=segment_ids,
            img_features=img_features
        )
        
        # Only keep last layer hidden states (no output attentions)
        encoded_layers = self.encoder(hidden_states=embedding_output,
                                      attention_mask=extended_attention_mask,
                                      head_mask=self.fixed_head_masks)
        sequence_output = encoded_layers[0]

        # Bert Pooling: take hidden state of the first token of sequence_output
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class FineTuneVisualBertModel(nn.Module):
    """ Explanation."""

    def __init__(self, bert_model_name, pretrained_params_file, visual_embedding_dim, num_labels):
        super(FineTuneVisualBertModel, self).__init__()
        self.bert_model_name = bert_model_name
        self.pretrained_params_file = pretrained_params_file
        self.visual_embedding_dim = visual_embedding_dim
        self.num_labels = num_labels

        self.state_dict = torch.load(pretrained_params_file, map_location=torch.device('cpu'))

        # Initialize VisualBertModel from pretrained
        self.bert_config = BertConfig.from_pretrained(self.bert_model_name, num_labels=self.num_labels)

        self.bert = VisualBertModel.from_pretrained(
                        pretrained_model_name_or_path=None,
                        config=self.bert_config,
                        state_dict=self.state_dict
                    )
        
        # Add layers for binary classification task
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert_config),
            nn.Linear(self.bert.config.hidden_size, self.num_labels)
        )

        # VisualBertModel initialized with pretrained weights

        # Initialize classifier randomly
        # self.classifier.apply(self.bert._init_weights)

        # Initialize classifier with finetuned weights
        self.classifier[0].dense.weight = nn.Parameter(self.state_dict['classifier.0.dense.weight'], requires_grad=True)
        self.classifier[0].dense.bias = nn.Parameter(self.state_dict['classifier.0.dense.bias'], requires_grad=True)
        self.classifier[0].LayerNorm.weight = nn.Parameter(self.state_dict['classifier.0.LayerNorm.weight'], requires_grad=True)
        self.classifier[0].LayerNorm.bias = nn.Parameter(self.state_dict['classifier.0.LayerNorm.bias'], requires_grad=True)
        self.classifier[1].weight = nn.Parameter(self.state_dict['classifier.1.weight'], requires_grad=True)
        self.classifier[1].bias = nn.Parameter(self.state_dict['classifier.1.bias'], requires_grad=True)

    def forward(
        self,
        input_ids,
        segment_ids,
        img_features,
        input_mask
    ):
        sequence_output, pooled_output = self.bert(
            input_ids,
            segment_ids,
            img_features,
            input_mask
        )

        output_dic = {}

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)

        output_dic["scores"] = reshaped_logits

        return output_dic

class PrepareVisualBertModel(nn.Module):
    """ Explanation."""

    def __init__(self, args):
        super(PrepareVisualBertModel, self).__init__()

        self.bert_model_name = 'bert-base-uncased'

        # self.fc7_w_file = os.path.join(args.params_path, 'fasterrcnn_fc7/fc7_w.pkl')
        # self.fc7_b_file = os.path.join(args.params_path, 'fasterrcnn_fc7/fc7_b.pkl')
        self.pretrained_params_file = os.path.join(args.params_path, 'visual_bert_finetuned/model.pth')

        self.visual_embedding_dim = 2048
        self.num_labels = 2

        # self.faster_rcnn_fc7 = FineTuneFasterRcnnFc7(weights_file=self.fc7_w_file, bias_file=self.fc7_b_file)
    
        self.model = FineTuneVisualBertModel(
            bert_model_name=self.bert_model_name,
            pretrained_params_file=self.pretrained_params_file,
            visual_embedding_dim=self.visual_embedding_dim,
            num_labels=self.num_labels
        )

    def forward(self, samples_batch):
        """ Make sure that every textual input has shape (batch_size, max_seq_length)
            and every visual inputs has shape (batch_size, img_features_number, img_features_dim). """

        # samples_batch["img_features"] = self.faster_rcnn_fc7(samples_batch["img_features"])

        output_dic = self.model(
            input_ids=samples_batch["input_ids"],
            segment_ids=samples_batch["segment_ids"],
            img_features=samples_batch["img_features"],
            input_mask=samples_batch["input_mask"]
        )

        return output_dic

        


