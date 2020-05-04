#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  DATA  ####################################
rm -rf ./data

download-gdrive 1kuLN2xqIKmb3U_-gYAdmxBiO4FS_T0ck cifar10-tiny.tar.gz
tar -xf cifar10-tiny.tar.gz &> /dev/null
mv ./cifar10-tiny ./data
rm cifar10-tiny.tar.gz


################################  pipeline 00  ################################
rm -rf ./logs


################################  pipeline 01  ################################
bash ./bin/catalyst-autolabel-pipeline.sh \
  --config-template ./configs/test_configs/autolabel.yml \
  --workdir ./logs \
  --datadir-clean ./data/clean \
  --datadir-raw ./data/raw \
  --n-trials 1 \
  --threshold 0.9 \
  --max-image-size 32  \
  --num-workers 0 \
  --batch-size 2

python -c """
from collections import Counter
import os
from pathlib import Path

prefix = Path('./logs/dataset_clean/images/')
counter = Counter()

for label_dir in prefix.glob('*'):
    for path in label_dir.glob('*.jpg'):
        basename = os.path.basename(path).replace('data_raw_', '')
        counter.update([basename])

num_correct = len([x for x in counter.values() if x == 2])
num_total = len(counter)
assert num_correct / num_total >= 0.8
"""


################################  pipeline 99  ################################
rm -rf ./logs
