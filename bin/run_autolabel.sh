#!/usr/bin/env bash

set -e

DATAPATH_RAW=""
DATAPATH_CLEAN=""
BASELOGDIR=""
N_TRIALS=10
THRESHOLD=0.95

if [[ -z "$RUN_CONFIG" ]]
then
      RUN_CONFIG=exp_splits.yml
fi


# bash argparse
while (( "$#" )); do
  case "$1" in
    --data-raw)
      DATAPATH_RAW=$2
      shift 2
      ;;
    --data-clean)
      DATAPATH_CLEAN=$2
      shift 2
      ;;
    --baselogdir)
      BASELOGDIR=$2
      shift 2
      ;;
    --n-trials)
      N_TRIALS=$2
      shift 2
      ;;
    --threshold)
      THRESHOLD=$2
      shift 2
      ;;
    *) # preserve positional arguments
      shift
      ;;
  esac
done

NUM_CLASSES=$(find "${DATAPATH_CLEAN}" -type d -maxdepth 1 | wc -l | awk '{print $1}')
NUM_CLASSES="$(($NUM_CLASSES-1))"
echo "NUM CLASSES: $NUM_CLASSES"

catalyst-data tag2label \
    --in-dir="${DATAPATH_RAW}" \
    --out-dataset="${DATAPATH_RAW}"/dataset_raw.csv \
    --out-labeling="${DATAPATH_RAW}"/tag2cls.json

for ((i=0; i < N_TRIALS; ++i)); do
    LOGDIR="${BASELOGDIR}/${i}"

    catalyst-data tag2label \
        --in-dir="${DATAPATH_CLEAN}" \
        --out-dataset="${DATAPATH_CLEAN}"/dataset_raw.csv \
        --out-labeling="${DATAPATH_CLEAN}"/tag2cls.json

    catalyst-data split-dataframe \
        --in-csv="${DATAPATH_CLEAN}"/dataset_raw.csv \
        --tag2class="${DATAPATH_CLEAN}"/tag2cls.json \
        --tag-column=tag \
        --class-column=class \
        --n-folds=5 \
        --train-folds=0,1,2,3 \
        --out-csv="${DATAPATH_CLEAN}"/dataset.csv

    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i ".bak" "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" "./configs/$RUN_CONFIG"
    elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
        sed -i "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" "./configs/$RUN_CONFIG"
    fi

    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir="${LOGDIR}" \
        --out_dir="${LOGDIR}":str \
        --stages/data_params/datapath="${DATAPATH_CLEAN}":str \
        --stages/data_params/in_csv_train="${DATAPATH_CLEAN}/dataset_train.csv":str \
        --stages/data_params/in_csv_valid="${DATAPATH_CLEAN}/dataset_valid.csv":str \
        --stages/infer/data_params/datapath="${DATAPATH_RAW}":str \
        --stages/infer/data_params/in_csv_train=None:str \
        --stages/infer/data_params/in_csv_valid=None:str \
        --stages/infer/data_params/in_csv_infer="${DATAPATH_RAW}/dataset_raw.csv":str

    PYTHONPATH=. python ./scripts/predictions2labels.py \
        --in-npy="${LOGDIR}"/predictions/infer.logits.npy \
        --in-csv-infer="${DATAPATH_RAW}"/dataset_raw.csv \
        --in-csv-train="${DATAPATH_CLEAN}"/dataset.csv \
        --in-tag2cls="${DATAPATH_CLEAN}"/tag2cls.json \
        --in-dir="${DATAPATH_RAW}" \
        --out-dir="${DATAPATH_CLEAN}"/ \
        --threshold="${THRESHOLD}"
done
