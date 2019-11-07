#!/usr/bin/env bash
set -e

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

mkdir -p data

gdrive_download 1czneYKcE2sT8dAMHz3FL12hOU7m1ZkE7 ants_bees_cleared_190806.tar.gz
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
CONFIG_TEMPLATE=./configs/templates/main.yml \
NUM_WORKERS=0 \
BATCH_SIZE=64 \
CRITERION=CrossEntropyLoss \
./bin/catalyst-classification-pipeline.sh --check


python -c """
import pathlib
from safitty import Safict

folder = list(pathlib.Path('./logs/').glob('logdir-*'))[0]
metrics = Safict.load(f'{folder}/checkpoints/_metrics.json')

loss_class = metrics.get('best', 'loss_class')
auc_class = metrics.get('best', 'auc_class/_mean')
accuracy_class01 = metrics.get('best', 'accuracy_class01')

print(loss_class)
print(auc_class)
print(accuracy_class01)

assert loss_class < 0.5
assert auc_class > 0.85
assert accuracy_class01 > 75
"""
