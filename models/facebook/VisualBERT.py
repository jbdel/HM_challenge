import os
from copy import deepcopy

import torch
import mmf
from mmf.models import BaseModel
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from mmf.models.visual_bert import VisualBERTForClassification


class VisualBERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/models/visual_bert/pretrain.yaml"

    def build(self):

        self.model = VisualBERTForClassification(self.config)
        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def flatten(self, sample_list, to_be_flattened=None, to_be_flattened_dim=None):
        if to_be_flattened is None:
            to_be_flattened = {}
        if to_be_flattened_dim is None:
            to_be_flattened_dim = {}
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])

        if sample_list.visual_embeddings_type is None:
            if sample_list.image_mask is not None:
                sample_list.visual_embeddings_type = torch.zeros_like(
                    sample_list.image_mask, dtype=torch.long
                )

        if sample_list.image_mask is not None:
            attention_mask = torch.cat(
                (sample_list.input_mask, sample_list.image_mask), dim=-1
            )
            if sample_list.masked_lm_labels is not None:
                assert sample_list.masked_lm_labels.size(
                    -1
                ) == sample_list.input_mask.size(-1)
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = sample_list.masked_lm_labels.size()
                assert len(size_masked_lm_labels) == 2
                new_lm_labels[
                    : size_masked_lm_labels[0], : size_masked_lm_labels[1]
                ] = sample_list.masked_lm_labels
                sample_list.masked_lm_labels = new_lm_labels
        else:
            attention_mask = sample_list.input_mask

        sample_list.attention_mask = attention_mask

        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list, **kwargs):
        to_be_flattened = [
            "input_ids",
            "token_type_ids",
            "input_mask",
            "image_mask",
            "masked_lm_labels",
            "position_embeddings_visual",
            "visual_embeddings_type",
        ]
        to_be_flattened_dim = ["image_text_alignment", "visual_embeddings"]

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if self.config.training_head_type == "nlvr2":
            bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
            bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
            bert_input_type_ids = torch.cat([bert_input_type_ids, bert_input_type_ids])

            # image input
            img0 = getattr(sample_list, "img0", {})
            image_info = getattr(img0, "image_info_0", {})
            image_dim_variable_0 = getattr(image_info, "max_features", None)
            image_feat_variable_0 = getattr(img0, "image_feature_0", None)

            img1 = getattr(sample_list, "img1", {})
            image_info = getattr(img1, "image_info_0", {})
            image_dim_variable_1 = getattr(image_info, "max_features", None)
            image_feat_variable_1 = getattr(img1, "image_feature_0", None)

            image_feat_variable = torch.cat(
                [image_feat_variable_0, image_feat_variable_1]
            )
            image_dim_variable = torch.cat([image_dim_variable_0, image_dim_variable_1])
        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feat_variable = getattr(sample_list, "image_feature_0", None)

        sample_list.visual_embeddings = image_feat_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids
        return sample_list

    def add_custom_params(self, sample_list):
        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = (
                torch.arange(visual_embeddings.size(-2))
                .expand(*visual_embeddings.size()[:-1])
                .cuda()
            )
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None

        return sample_list

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def forward(self, sample_list):
        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)

        output_dict = self.model(
            sample_list.input_ids,
            sample_list.input_mask,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual,
            sample_list.visual_embeddings_type,
            sample_list.image_text_alignment,
            sample_list.masked_lm_labels,
        )

        if "pretraining" in self.config.training_head_type:
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )

        return output_dict
