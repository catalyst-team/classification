from typing import Dict
import json
import collections
import numpy as np

import torch
import torch.nn as nn

from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose
from catalyst.data.augmentor import Augmentor
from catalyst.utils.parse import read_csv_data
from catalyst.dl.experiments import ConfigExperiment
from catalyst.data.dataset import ListDataset
from catalyst.data.sampler import BalanceClassSampler
from .transforms import train_transform, valid_transform


class Experiment(ConfigExperiment):

    def _prepare_logdir(self, config: Dict):
        model_params = config["model_params"]
        data_params = config["stages"]["data_params"]

        if data_params.get("train_folds") is not None:
            train_folds = "-".join(list(map(str, data_params["train_folds"])))
        else:
            train_folds = "split"
        hiddens = "-".join(list(map(
            str, model_params["embedding_net_params"]["hiddens"])))
        return f"{train_folds}" \
            f".{model_params['model']}" \
            f".{model_params['encoder_params']['arch']}" \
            f".{model_params['encoder_params']['pooling']}" \
            f".{hiddens}"

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            for param in model_.encoder_net.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model_.encoder_net.parameters():
                param.requires_grad = True
        return model_

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None, image_size=224):
        if mode == "train":
            transform_fn = train_transform(image_size=image_size)
        elif mode in ["valid", "infer"]:
            transform_fn = valid_transform(image_size=image_size)
        else:
            raise NotImplementedError

        return Augmentor(
            dict_key="image",
            augment_fn=lambda x: transform_fn(image=x)["image"]
        )

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
        train_folds: str = None,
        valid_folds: str = None,
        tag2class: str = None,
        class_column: str = None,
        tag_column: str = None,
        folds_seed: int = 42,
        n_folds: int = 5,
        one_hot_classes: int = None,
        image_size: int = 224
    ):
        datasets = collections.OrderedDict()
        tag2class = json.load(open(tag2class)) \
            if tag2class is not None \
            else None

        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=folds_seed,
            n_folds=n_folds
        )

        open_fn = [
            ImageReader(
                input_key="filepath", output_key="image", datapath=datapath
            ),
            ScalarReader(
                input_key="class",
                output_key="targets",
                default_value=-1,
                dtype=np.int64
            )
        ]

        if one_hot_classes:
            open_fn.append(
                ScalarReader(
                    input_key="class",
                    output_key="targets_one_hot",
                    default_value=-1,
                    dtype=np.int64,
                    one_hot_classes=one_hot_classes
                ))

        open_fn = ReaderCompose(readers=open_fn)

        for source, mode in zip(
                (df_train, df_valid, df_infer),
                ("train", "valid", "infer")):
            if len(source) > 0:
                dataset = ListDataset(
                    source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(
                        stage=stage, mode=mode, image_size=image_size
                    ),
                )
                if mode == "train":
                    labels = [x["class"] for x in source]
                    sampler = BalanceClassSampler(labels, mode="upsampling")
                    dataset = {
                        "dataset": dataset,
                        "sampler": sampler
                    }
                datasets[mode] = dataset

        return datasets
