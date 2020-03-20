#!/usr/bin/env bash

set -e

usage()
{
  cat << USAGE >&2
Usage: $(basename "$0") [OPTION...] [catalyst-dl run args...]

  --n-trials N_TRIALS                 Number of trials to repeat train / infer process
  --threshold THRESHOLD               Threshold for prediction confidence in labeling
  -j, --num-workers NUM_WORKERS       Number of data loading/processing workers
  --max-image-size MAX_IMAGE_SIZE     Target size of images e.g. 224, 448
  --config-template CONFIG_TEMPLATE   Model config to use
  --datadir-clean DATADIR_CLEAN       Path to root directory with labeled data
  --datadir-raw DATADIR_RAW           Path to root directory with unlabeled data
  --workdir WORKDIR                   Working directory, where to store the result
  catalyst-dl run args                Execute \`catalyst-dl run\` with args

Example:
  CUDA_VISIBLE_DEVICES=0 \\
  CUDNN_BENCHMARK="True" \\
  CUDNN_DETERMINISTIC="True" \\
  ./bin/catalyst-autolabel-pipeline.sh \\
    --n-trials 10 \\
    --threshold 0.95 \\
    --num-workers 4 \\
    --max-image-size 224 \\
    --config-template ./configs/templates/autolabel.yml \\
    --datadir-clean ./data/origin \\
    --datadir-raw ./data/raw \\
    --workdir ./logs \\
    --verbose

USAGE
  exit 1
}


# ---- environment variables

N_TRIALS=${N_TRIALS:=10}
THRESHOLD=${THRESHOLD:=0.95}
NUM_WORKERS=${NUM_WORKERS:=4}
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:=224}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE:="./configs/templates/autolabel.yml"}
DATADIR_CLEAN=${DATADIR_CLEAN:="./data/origin"}
DATADIR_RAW=${DATADIR_RAW:="./data/raw"}
WORKDIR=${WORKDIR:="./logs"}

_run_args=""
while (( "$#" )); do
  case "$1" in
    --n-trials)
      N_TRIALS=$2
      shift 2
      ;;
    --threshold)
      THRESHOLD=$2
      shift 2
      ;;
    -j|--num-workers)
      NUM_WORKERS=$2
      shift 2
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE=$2
      shift 2
      ;;
    --config-template)
      CONFIG_TEMPLATE=$2
      shift 2
      ;;
    --datadir-clean)
      DATADIR_CLEAN=$2
      shift 2
      ;;
    --datadir-raw)
      DATADIR_RAW=$2
      shift 2
      ;;
    --workdir)
      WORKDIR=$2
      shift 2
      ;;
    *)
      _run_args="${_run_args} $1"
      shift
      ;;
  esac
done

export RAW_DATASET_DIR=${WORKDIR}/dataset_raw
export CLEAN_DATASET_DIR=${WORKDIR}/dataset_clean


# ---- data preparation

# process raw & clean data
#  to avoid the repetition of data processing inside the loop
bash ./bin/_data_preparation.sh \
  --num-workers ${NUM_WORKERS} \
  --max-image-size ${MAX_IMAGE_SIZE} \
  --datadir ${DATADIR_RAW} \
  --dataset-dir ${RAW_DATASET_DIR} \
  --images-dir ${RAW_DATASET_DIR}/images

# process clean data without splitting it,
#  as folds will be prepared at the start of each round
bash ./bin/_data_preparation.sh --no-split \
  --num-workers ${NUM_WORKERS} \
  --max-image-size ${MAX_IMAGE_SIZE} \
  --datadir ${DATADIR_CLEAN} \
  --dataset-dir ${CLEAN_DATASET_DIR} \
  --images-dir ${CLEAN_DATASET_DIR}/images


# ---- raw data annotation through pseudo-labeling

for ((i=0; i < N_TRIALS; ++i)); do
  # split clean and pseudo-labeled data into folds
  bash ./bin/_data_preparation.sh --no-process \
    --dataset-dir ${CLEAN_DATASET_DIR} \
    --images-dir ${CLEAN_DATASET_DIR}/images

  # train model on clean and pseudo-labeled data
  bash ./bin/catalyst-classification-pipeline.sh \
    --dataset-dir ${CLEAN_DATASET_DIR} \
    --config-template ${CONFIG_TEMPLATE} \
    --stages/infer="None":str \
    -s ${_run_args}

  # search for logdir of the last run
  LOGDIR=$(ls -d ${WORKDIR}/logdir-* | sort -nr | head -1)

  # make predictions for the raw data
  catalyst-dl run \
    -C ${LOGDIR}/configs/_common.yml ${LOGDIR}/configs/config.yml \
    --stages/callbacks_params="{}":dict \
    --stages/stage1="None":str --stages/stage2="None":str \
    --stages/infer/data_params/datapath="${RAW_DATASET_DIR}/images":str \
    --stages/infer/data_params/in_csv_infer="${RAW_DATASET_DIR}/dataset_raw.csv":str \
    --stages/infer/callbacks_params/infer/out_prefix="${LOGDIR}/predictions/":str

  # select the most confident predictions and join them to the clear data
  python ./scripts/predictions2labels.py \
    --in-npy="${LOGDIR}"/predictions/infer.logits.npy \
    --in-csv-infer=${RAW_DATASET_DIR}/dataset_raw.csv \
    --in-csv-train=${CLEAN_DATASET_DIR}/dataset.csv \
    --in-tag2cls=${CLEAN_DATASET_DIR}/tag2class.json \
    --in-dir=${RAW_DATASET_DIR}/images \
    --out-dir=${CLEAN_DATASET_DIR}/images \
    --threshold=${THRESHOLD}
done
