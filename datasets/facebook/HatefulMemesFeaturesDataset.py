import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.configuration import get_global_config, get_mmf_env, get_zoo_config
from mmf.utils.general import get_absolute_path
import os, collections
import mmf.utils.download as download
from omegaconf import OmegaConf

class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        self.build(config, dataset_name)
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        ), "config's 'use_images' must be true to use image dataset"
        self.init_processors()

    def preprocess_sample_info(self, sample_info):
        image_path = sample_info["img"]
        # img/02345.png -> 02345
        feature_path = image_path.split("/")[-1].split(".")[0]
        # Add feature_path key for feature_database access
        sample_info["feature_path"] = f"{feature_path}.npy"
        return sample_info

    def build(self, config, dataset_name):
        self._download_requirement(config, dataset_name, 'defaults')

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]

        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)

        # Instead of using idx directly here, use sample_info to fetch
        # the features as feature_path has been dynamically added
        features = self.features_db.get(sample_info)
        if hasattr(self, "transformer_bbox_processor"):
            features["image_info_0"] = self.transformer_bbox_processor(
                features["image_info_0"]
            )
        current_sample.update(features)

        if "label" in sample_info:
            current_sample.targets = torch.tensor(
                sample_info["label"], dtype=torch.long
            )
        current_sample.dataset_name = self._dataset_name
        current_sample.dataset_type = self._dataset_type
        return current_sample

    def _download_requirement(
        self, config, requirement_key, requirement_variation="defaults"
    ):

        version, resources = get_zoo_config(
            requirement_key, requirement_variation, get_global_config("env.dataset_zoo"), "datasets"
        )

        if resources is None:
            return

        requirement_split = requirement_key.split(".")
        dataset_name = requirement_split[0]

        # The dataset variation has been directly passed in the key so use it instead
        if len(requirement_split) >= 2:
            dataset_variation = requirement_split[1]
        else:
            dataset_variation = requirement_variation
        # We want to use root env data_dir so that we don't mix up our download
        # root dir with the dataset ones

        download_path = os.path.join(
            OmegaConf.select(config, "data_dir"), "datasets", dataset_name, dataset_variation
        )
        download_path = get_absolute_path(download_path)
        if not isinstance(resources, collections.abc.Mapping):
            download.download_resources(resources, download_path, version)
        else:
            use_features = config.get("use_features", False)
            use_images = config.get("use_images", False)

            if use_features:
                self._download_based_on_attribute(
                    resources, download_path, version, "features"
                )

            if use_images:
                self._download_based_on_attribute(
                    resources, download_path, version, "images"
                )

            self._download_based_on_attribute(
                resources, download_path, version, "annotations"
            )
            download.download_resources(
                resources.get("extras", []), download_path, version
            )

    def _download_based_on_attribute(
        self, resources, download_path, version, attribute
    ):
        path = os.path.join(download_path, attribute)
        download.download_resources(resources.get(attribute, []), path, version)
