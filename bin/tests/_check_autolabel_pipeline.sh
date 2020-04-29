#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

###################################  DATA  ####################################
rm -rf ./data

mkdir -p ./data/clean
mkdir -p ./data/raw/all

cp -r ./dataset/* ./data/clean

for FILE in ./data/clean/*/*.jpg; do
  BASENAME=$(basename "${FILE}")
  cp "${FILE}" ./data/raw/all/data_raw_"${BASENAME}"
done

################################  pipeline 00  ################################
rm -rf ./logs

################################  pipeline 01  ################################
bash ./bin/catalyst-autolabel-pipeline.sh \
  --workdir ./logs \
  --datadir-clean ./data/clean \
  --datadir-raw ./data/raw \
  --n-trials 1 \
  --threshold 0.8 \
  --config-template ./configs/templates/autolabel.yml \
  --max-image-size 224 \
  --num-workers 4 \
  --batch-size 256

python -c """
from collections import Counter
import os
from pathlib import Path


prefix = Path('./logs/dataset_clean/images/')
for label_dir in prefix.glob('*'):
    counter = Counter()

    for path in label_dir.glob('*.jpg'):
        basename = os.path.basename(path).replace('data_raw_', '')
        counter.update([basename])

    assert all(map(lambda x: x == 2, counter.values()))
"""


################################  pipeline 99  ################################
rm -rf ./logs

