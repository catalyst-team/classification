from typing import Optional
import collections

import numpy as np
import safitty

import torch
import torch.nn as nn

from catalyst.contrib.data.cv import ImageReader
from catalyst.data import (
    BalanceClassSampler,
    ListDataset,
    ReaderCompose,
    ScalarReader,
)
from catalyst.dl import ConfigExperiment
from catalyst.utils import read_csv_data


class Experiment(ConfigExperiment):
    """Classification Experiment."""

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )

        if stage in ["debug", "stage1"]:
            for param in model.encoder_net.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model.encoder_net.parameters():
                param.requires_grad = True
        return model

    def get_datasets(
        self,
        stage: str,
        datapath: Optional[str] = None,
        in_csv: Optional[str] = None,
        in_csv_train: Optional[str] = None,
        in_csv_valid: Optional[str] = None,
        in_csv_infer: Optional[str] = None,
        train_folds: Optional[str] = None,
        valid_folds: Optional[str] = None,
        tag2class: Optional[str] = None,
        class_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        folds_seed: int = 42,
        n_folds: int = 5,
        one_hot_classes: Optional[int] = None,
        balance_strategy: str = "upsampling",
    ):
        """Returns the datasets for a given stage and epoch.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            datapath (str): path to folder with images and masks
            in_csv (Optional[str]): path to CSV annotation file. Look at
                :func:`catalyst.contrib.utils.pandas.read_csv_data` for details
            in_csv_train (Optional[str]): path to CSV annotaion file
                with train samples.
            in_csv_valid (Optional[str]): path to CSV annotaion file
                with the validation samples
            in_csv_infer (Optional[str]): path to CSV annotaion file
                with test samples
            train_folds (Optional[str]): folds to use for training
            valid_folds (Optional[str]): folds to use for validation
            tag2class (Optional[str]): path to JSON file with mapping from
                class name (tag) to index
            class_column (Optional[str]): name of class index column in the CSV
            tag_column (Optional[str]): name of class name in the CSV file
            folds_seed (int): random seed to use
            n_folds (int): number of folds on which data will be split
            one_hot_classes (int): number of one-hot classes
            balance_strategy (str): strategy to handle imbalanced data,
                look at :class:`catalyst.data.BalanceClassSampler` for details

        Returns:
            Dict: dictionary with datasets for current stage.
        """
        datasets = collections.OrderedDict()
        tag2class = safitty.load(tag2class) if tag2class is not None else None

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
            n_folds=n_folds,
        )

        open_fn = [
            ImageReader(
                input_key="filepath", output_key="image", rootpath=datapath
            ),
            ScalarReader(
                input_key="class",
                output_key="targets",
                default_value=-1,
                dtype=np.int64,
            ),
        ]

        if one_hot_classes:
            open_fn.append(
                ScalarReader(
                    input_key="class",
                    output_key="targets_one_hot",
                    default_value=-1,
                    dtype=np.int64,
                    one_hot_classes=one_hot_classes,
                )
            )

        open_fn = ReaderCompose(readers=open_fn)

        for source, mode in zip(
            (df_train, df_valid, df_infer), ("train", "valid", "infer")
        ):
            if source is not None and len(source) > 0:
                dataset = ListDataset(
                    source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(
                        stage=stage, dataset=mode
                    ),
                )
                if mode == "train":
                    labels = [x["class"] for x in source]
                    sampler = BalanceClassSampler(
                        labels, mode=balance_strategy
                    )
                    dataset = {"dataset": dataset, "sampler": sampler}
                datasets[mode] = dataset

        return datasets
