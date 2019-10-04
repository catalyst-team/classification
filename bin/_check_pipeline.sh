#!/usr/bin/env bash
set -e

mkdir -p data

wget https://www.dropbox.com/s/8aiufmo0yyq3cf3/ants_bees_cleared_190806.tar.gz
tar -xf ants_bees_cleared_190806.tar.gz &>/dev/null
mv ants_bees_cleared_190806 ./data/origin

USE_WANDB=0 \
CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
WORKDIR=./logs \
DATADIR=./data/origin \
MAX_IMAGE_SIZE=224 \
BALANCE_STRATEGY=64 \
CONFIG_TEMPLATE=./configs/templates/ce.yml \
NUM_WORKERS=0 \
BATCH_SIZE=64 \
./bin/catalyst-classification-pipeline.sh --check


python -c """
import pathlib
from safitty import Safict

folder = list(pathlib.Path('./logs/').glob('logdir-*'))[0]
metrics = metrics=Safict.load(f'{folder}/checkpoints/_metrics.json')
assert metrics.get('best', 'loss_class') < 0.5
assert metrics.get('best', 'auc_class/_mean') > 0.85
assert metrics.get('best', 'accuracy_class01') > 80
"""
