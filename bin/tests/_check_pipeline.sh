#!/usr/bin/env bash
set -e

mkdir -p ./data

download-gdrive 1czneYKcE2sT8dAMHz3FL12hOU7m1ZkE7 ants_bees_cleared_190806.tar.gz
tar -xf ants_bees_cleared_190806.tar.gz &>/dev/null
mv ants_bees_cleared_190806 ./data/origin

CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
./bin/catalyst-classification-pipeline.sh \
  --config-template ./configs/templates/main.yml \
  --workdir ./logs \
  --datadir ./data/origin \
  --num-workers 0 \
  --batch-size 64 \
  --max-image-size 224 \
  --balance-strategy 64 \
  --criterion CrossEntropyLoss \
  --check


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

assert loss_class < 0.6
assert auc_class > 0.8
assert accuracy_class01 > 70
"""
