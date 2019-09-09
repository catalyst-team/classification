#!/usr/bin/env bash

wget -P ./data https://www.dropbox.com/s/eeme52kwnvz255d/mnist.tar.gz
tar -xvf ./data/mnist.tar.gz -C ./data
mv ./data/mnist ./data/dataset

catalyst-data tag2label \
  --in-dir=./data/dataset \
  --out-dataset=./data/dataset_raw.csv \
  --out-labeling=./data/tag2cls.json
catalyst-data split-dataframe \
  --in-csv=./data/dataset_raw.csv \
  --tag2class=./data/tag2cls.json \
  --tag-column=tag \
  --class-column=class \
  --n-folds=5 \
  --train-folds=0,1,2,3 \
  --out-csv=./data/dataset.csv

export NUM_CLASSES=10; bash ./bin/prepare_configs.sh

catalyst-dl run \
    --config=configs/exp_splits.yml --check \
    --stages/stage2=None:str \
    --stages/infer=None:str
python -c """
data = open('./logs/classification/metrics.txt', 'r').readlines()
assert float(data[8].rsplit('embeddings_loss=', 1)[-1][:6]) < float(data[1].rsplit('embeddings_loss=', 1)[-1][:6])
assert float(data[8].rsplit('embeddings_loss=', 1)[-1][:6]) < 2.2
"""

catalyst-dl run \
    --config=configs/exp_splits_bce.yml --check \
    --stages/stage2=None:str \
    --stages/infer=None:str
python -c """
data = open('./logs/classification_bce/metrics.txt', 'r').readlines()
assert float(data[8].rsplit('embeddings_loss=', 1)[-1][:6]) < float(data[1].rsplit('embeddings_loss=', 1)[-1][:6])
assert float(data[8].rsplit('embeddings_loss=', 1)[-1][:6]) < 0.55
"""

catalyst-dl run \
    --config=configs/exp_splits_focal.yml --check \
    --stages/stage2=None:str \
    --stages/infer=None:str
python -c """
data = open('./logs/classification_focal/metrics.txt', 'r').readlines()
assert float(data[8].rsplit('embeddings_loss=', 1)[-1][:6]) < float(data[1].rsplit('embeddings_loss=', 1)[-1][:6])
assert float(data[8].rsplit('embeddings_loss=', 1)[-1][:6]) < 0.55
"""

catalyst-dl run \
    --config=configs/exp_splits_augs.yml --check  \
    --stages/stage2=None:str \
    --stages/infer=None:str
python -c """
data = open('./logs/classification_augs/metrics.txt', 'r').readlines()
assert float(data[8].rsplit('loss_class_rotation=', 1)[-1][:6]) < float(data[1].rsplit('loss_class_rotation=', 1)[-1][:6])
assert float(data[8].rsplit('loss_class_rotation=', 1)[-1][:6]) < 4.4
"""
