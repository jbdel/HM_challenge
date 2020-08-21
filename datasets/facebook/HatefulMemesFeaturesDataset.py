import os
import numpy as np
import torch
import lmdb
import pickle
import os
import numpy as np
import torch
import omegaconf

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset



class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        ), "config's 'use_images' must be true to use image dataset"

    def preprocess_sample_info(self, sample_info):
        image_path = sample_info["img"]
        # img/02345.png -> 02345
        feature_path = image_path.split("/")[-1].split(".")[0]
        # Add feature_path key for feature_database access
        sample_info["feature_path"] = f"{feature_path}.npy"
        return sample_info

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


class FeatureReader:
    def __init__(self, base_path, depth_first, max_features=None):
        """Feature Reader class for reading features.

        Note: Deprecation: ndim and image_feature will be deprecated later
        and the format will be standardize using features from detectron.

        Parameters
        ----------
        ndim : int
            Number of expected dimensions in features
        depth_first : bool
            CHW vs HWC
        max_features : int
            Number of maximum bboxes to keep

        Returns
        -------
        type
            Description of returned object.

        """
        self.base_path = base_path
        ndim = None
        self.feat_reader = None
        self.depth_first = depth_first
        self.max_features = max_features
        self.ndim = ndim
        self._init_reader()


    def _init_reader(self):
        # Currently all lmdb features are with ndim == 2
        if self.base_path.endswith(".lmdb"):
            self.feat_reader = LMDBFeatureReader(self.max_features, self.base_path)
        else:
            raise TypeError("unknown image feature format")

    def read(self, image_feat_path):
        image_feat_path = os.path.join(self.base_path, image_feat_path)
        return self.feat_reader.read(image_feat_path)


class PaddedFasterRCNNFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc
        self.first = True
        self.take_item = False

    def _load(self, image_feat_path):
        image_info = {}
        image_info["features"] = np.load(image_feat_path, allow_pickle=True)

        info_path = "{}_info.npy".format(image_feat_path.split(".npy")[0])
        if os.path.exists(info_path):
            image_info.update(np.load(info_path, allow_pickle=True).item())

        return image_info

    def read(self, image_feat_path):
        image_info = self._load(image_feat_path)
        if self.first:
            self.first = False
            if (
                image_info["features"].size == 1
                and "image_feat" in image_info["features"].item()
            ):
                self.take_item = True

        image_feature = image_info["features"]

        if self.take_item:
            item = image_info["features"].item()
            if "image_text" in item:
                image_info["image_text"] = item["image_text"]
                image_info["is_ocr"] = item["image_bbox_source"]
                image_feature = item["image_feat"]

            if "info" in item:
                if "image_text" in item["info"]:
                    image_info.update(item["info"])
                image_feature = item["feature"]

        # Handle the case of ResNet152 features
        if len(image_feature.shape) > 2:
            shape = image_feature.shape
            image_feature = image_feature.reshape(-1, shape[-1])

        image_loc, image_dim = image_feature.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc,] = image_feature[: self.max_loc, :]  # noqa
        image_feature = torch.from_numpy(tmp_image_feat)

        del image_info["features"]
        image_info["max_features"] = torch.tensor(image_loc, dtype=torch.long)
        return image_feature, image_info


class LMDBFeatureReader(PaddedFasterRCNNFeatureReader):
    def __init__(self, max_loc, base_path):
        super().__init__(max_loc)
        self.db_path = base_path

        if not os.path.exists(self.db_path):
            raise RuntimeError(
                "{} path specified for LMDB features doesn't exists.".format(
                    self.db_path
                )
            )
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False, buffers=True) as txn:
            self.image_ids = pickle.loads(txn.get(b"keys"))
            self.image_id_indices = {
                self.image_ids[i]: i for i in range(0, len(self.image_ids))
            }

    def _load(self, image_file_path):
        if self.env is None:
            self._init_db()

        split = os.path.relpath(image_file_path, self.db_path).split(".npy")[0]

        try:
            image_id = int(split.split("_")[-1])
            # Try fetching to see if it actually exists otherwise fall back to
            # default
            img_id_idx = self.image_id_indices[str(image_id).encode()]
        except (ValueError, KeyError):
            # The image id is complex or involves folder, use it directly
            image_id = str(split).encode()
            img_id_idx = self.image_id_indices[image_id]

        with self.env.begin(write=False, buffers=True) as txn:
            image_info = pickle.loads(txn.get(self.image_ids[img_id_idx]))

        return image_info



if __name__ == "__main__":
    config = {'data_dir': '/home/jb/.cache/torch/mmf/data/datasets',
          'depth_first': False,
          'fast_read': False,
          'use_images': False,
          'use_features': True,
          'images': {'train': ['hateful_memes/defaults/images/'],
            'val': ['hateful_memes/defaults/images/'],
            'test': ['hateful_memes/defaults/images/']},
            'features': {'train': ['hateful_memes/defaults/features/detectron.lmdb'],
            'val': ['hateful_memes/defaults/features/detectron.lmdb'],
            'test': ['hateful_memes/defaults/features/detectron.lmdb']},
            'annotations': {'train': ['hateful_memes/defaults/annotations/train.jsonl'],
            'val': ['hateful_memes/defaults/annotations/dev.jsonl'],
            'test': ['hateful_memes/defaults/annotations/test.jsonl']},
            'max_features': 100,
            'processors': {'text_processor': {'type': 'bert_tokenizer',
            'params': {'max_length': 14,
            'vocab': {'type': 'intersected',
            'embedding_name': 'glove.6B.300d',
            'vocab_file': 'hateful_memes/defaults/extras/vocabs/vocabulary_100k.txt'},
            'preprocessor': {'type': 'simple_sentence',
            'params': {}},
            'tokenizer_config': {'type': 'bert-base-uncased',
            'params': {'do_lower_case': True}},
            'mask_probability': 0,
            'max_seq_length': 128}},
            'bbox_processor': {'type': 'bbox', 'params': {'max_length': 50}},
            'image_processor': {'type': 'torchvision_transforms',
            'params': {'transforms': [{'type': 'Resize',
            'params': {'size': [256, 256]}},
                                      {'type': 'CenterCrop', 'params': {'size': [224, 224]}}, 'ToTensor',
                                    'GrayScaleTo3Channels',
                                    {'type': 'Normalize',
                                    'params': {'mean': [0.46777044,
                                    0.44531429, 0.40661017],
                                    'std': [0.12221994, 0.12145835, 0.14380469]}}]}}},
                                    'return_features_info': True}

    F = FeatureReader("/home/jb/.cache/torch/mmf/data/datasets/hateful_memes/defaults/features/detectron.lmdb", False, max_features=100)
    image_feature, image_info = F.read("18640.npy")

    x =  HatefulMemesFeaturesDataset(omegaconf.dictconfig.DictConfig(config))
