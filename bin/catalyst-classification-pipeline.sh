#!/usr/bin/env bash
#title           :catalyst-classification-pipeline
#description     :catalyst.dl script for full classification pipeline run
#author          :Sergey Kolesnikov
#author_email    :scitator@gmail.com
#date            :20190909
#version         :19.09.2
#==============================================================================

set -e

usage()
{
  cat << USAGE >&2
Usage: $(basename "$0") [OPTION...] [catalyst-dl run args...]

  -s, --skipdata                       Skip data preparation
  -j, --num-workers NUM_WORKERS        Number of data loading/processing workers
  -b, --batch-size BATCH_SIZE          Mini-batch size
  --max-image-size MAX_IMAGE_SIZE      Target size of images e.g. 224, 448
  --balance-strategy BALANCE_STRATEGY  Images in epoch per class, e.g. 1024
  -m, --traced-method TRACED_METHOD    Model method to trace
  --config-template CONFIG_TEMPLATE    Model config to use
  --criterion CRITERION                Criterion to use
  --datadir DATADIR
  --workdir WORKDIR
  catalyst-dl run args                 Execute \`catalyst-dl run\` with args

Example:
  CUDA_VISIBLE_DEVICES=0 \\
  CUDNN_BENCHMARK="True" \\
  CUDNN_DETERMINISTIC="True" \\
  ./bin/catalyst-classification-pipeline.sh \\
    --workdir ./logs \\
    --datadir ./data/origin \\
    --max-image-size 224 \\
    --balance-strategy 256 \\
    --config-template ./configs/templates/main.yml \\
    --num-workers 4 \\
    --batch-size 256 \\
    --criterion FocalLossMultiClass
USAGE
  exit 1
}


# ---- environment variables

NUM_WORKERS=${NUM_WORKERS:=4}
BATCH_SIZE=${BATCH_SIZE:=64}
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:=224}
BALANCE_STRATEGY=${BALANCE_STRATEGY:="null"}
TRACED_METHOD=${TRACED_METHOD:="forward_class"}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE:="./configs/templates/main.yml"}
CRITERION=${CRITERION:="FocalLossMultiClass"}  # BCEWithLogits , CrossEntropyLoss
DATADIR=${DATADIR:="./data/origin"}
WORKDIR=${WORKDIR:="./logs"}
DATASET_DIR=${DATASET_DIR:=${WORKDIR}/dataset}
SKIPDATA=""

_run_args=""
while (( "$#" )); do
  case "$1" in
    -j|--num-workers)
      NUM_WORKERS=$2
      shift 2
      ;;
    -b|--batch-size)
      BATCH_SIZE=$2
      shift 2
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE=$2
      shift 2
      ;;
    --balance-strategy)
      BALANCE_STRATEGY=$2
      shift 2
      ;;
    -m|--traced-method)
      TRACED_METHOD=$2
      shift 2
      ;;
    --config-template)
      CONFIG_TEMPLATE=$2
      shift 2
      ;;
    --criterion)
      CRITERION=$2
      shift 2
      ;;
    --datadir)
      DATADIR=$2
      shift 2
      ;;
    --workdir)
      WORKDIR=$2
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR=$2
      shift 2
      ;;
    -s|--skipdata)
      SKIPDATA="true"
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      _run_args="${_run_args} $1"
      shift
      ;;
  esac
done

date=$(date +%y%m%d-%H%M%S)
postfix=$(openssl rand -hex 4)
logname="${date}-${postfix}"
export CONFIG_DIR=${WORKDIR}/configs-${logname}
export LOGDIR=${WORKDIR}/logdir-${logname}
export SERVING_DIR=${WORKDIR}/serving-${logname}

for dir in ${WORKDIR} ${DATASET_DIR} ${CONFIG_DIR} ${LOGDIR} ${SERVING_DIR}; do
  mkdir -p ${dir}
done


# ---- data preparation

if [[ -z "${SKIPDATA}" ]]; then
  bash ./bin/_data_preparation.sh \
    --num-workers ${NUM_WORKERS} \
    --max-image-size ${MAX_IMAGE_SIZE} \
    --datadir ${DATADIR} \
    --workdir ${WORKDIR} \
    --dataset-dir ${DATASET_DIR}
fi


# ---- config preparation

python ./scripts/prepare_config.py \
  --in-template=${CONFIG_TEMPLATE} \
  --out-config=${CONFIG_DIR}/config.yml \
  --expdir=./src \
  --dataset-path=${DATASET_DIR} \
  --num-workers=${NUM_WORKERS} \
  --batch-size=${BATCH_SIZE} \
  --max-image-size=${MAX_IMAGE_SIZE} \
  --balance-strategy=${BALANCE_STRATEGY} \
  --criterion=${CRITERION}

cp -r ./configs/_common.yml ${CONFIG_DIR}/_common.yml


# ---- model training

catalyst-dl run \
  -C ${CONFIG_DIR}/_common.yml ${CONFIG_DIR}/config.yml \
  --logdir ${LOGDIR} ${_run_args}


# ---- model tracing

catalyst-dl trace \
  ${LOGDIR} \
  --method ${TRACED_METHOD} \
  --out-model ${LOGDIR}/traced.pth


# ---- model serving

TAG2CLS_PATH=$(python << EOF
import json
from safitty import Safict
path = Safict.load("${LOGDIR}/configs/_config.json").get(
    "stages", "data_params", "datapath"
).replace("images", "")
print(path + "tag2class.json")
EOF
)

cp ${LOGDIR}/traced.pth ${SERVING_DIR}/model.pth
cp ${TAG2CLS_PATH} ${SERVING_DIR}
