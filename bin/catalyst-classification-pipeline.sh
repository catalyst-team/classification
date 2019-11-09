#!/usr/bin/env bash
#title           :catalyst-classification-pipeline
#description     :catalyst.dl script for full classification pipeline run
#author          :Sergey Kolesnikov
#author_email    :scitator@gmail.com
#date            :20190909
#version         :19.09.2
#==============================================================================

# usage:
# WORKDIR=/path/to/logdir \
# DATADIR=/path/to/dataset \
# MAX_IMAGE_SIZE=... \  # 224 or 448 works good
# BALANCE_STRATEGY=... \  # images in epoch per class, 1024 works good
# CONFIG_TEMPLATE=... \ # model config to use
# CRITERION=... \ # criterion
# ./bin/catalyst-classification-pipeline.sh

# example:
# CUDA_VISIBLE_DEVICES=0 \
# CUDNN_BENCHMARK="True" \
# CUDNN_DETERMINISTIC="True" \
# WORKDIR=./logs \
# DATADIR=./data/origin \
# MAX_IMAGE_SIZE=224 \  # 224 or 448 works good
# BALANCE_STRATEGY=256 \  # images in epoch per class, 1024 works good
# CONFIG_TEMPLATE=./configs/templates/main.yml \
# NUM_WORKERS=4 \
# BATCH_SIZE=256 \
# CRITERION=FocalLossMultiClass \
# ./bin/catalyst-classification-pipeline.sh

set -e

# --- test part
# uncomment and run bash ./bin/catalyst-classification-pipeline.sh

#mkdir -p ./data
#download-gdrive 1czneYKcE2sT8dAMHz3FL12hOU7m1ZkE7 ants_bees_cleared_190806.tar.gz
#tar -xf ants_bees_cleared_190806.tar.gz &>/dev/null
#mv ants_bees_cleared_190806 ./data/origin
#
#export CUDNN_BENCHMARK="True"
#export CUDNN_DETERMINISTIC="True"
#
#export CONFIG_TEMPLATE=./configs/templates/main.yml
#export WORKDIR=./logs
#export DATADIR=./data/origin
#export NUM_WORKERS=4
#export BATCH_SIZE=64
#export MAX_IMAGE_SIZE=128
#export BALANCE_STRATEGY=128
#export CRITERION=FocalLossMultiClass


# ---- environment variables

if [[ -z "$NUM_WORKERS" ]]; then
      NUM_WORKERS=4
fi

if [[ -z "$BATCH_SIZE" ]]; then
      BATCH_SIZE=64
fi

if [[ -z "$MAX_IMAGE_SIZE" ]]; then
      MAX_IMAGE_SIZE=224
fi

if [[ -z "$BALANCE_STRATEGY" ]]; then
      BALANCE_STRATEGY="null"
fi

if [[ -z "$TRACED_METHOD" ]]; then
      TRACED_METHOD="forward_class"
fi

if [[ -z "$CONFIG_TEMPLATE" ]]; then
      CONFIG_TEMPLATE="./configs/templates/main.yml"
fi

if [[ -z "$CRITERION" ]]; then
        CRITERION="FocalLossMultiClass" # BCEWithLogits , CrossEntropyLoss
fi

if [[ -z "$DATADIR" ]]; then
      DATADIR="./data/origin"
fi

if [[ -z "$WORKDIR" ]]; then
      WORKDIR="./logs"
fi

SKIPDATA=""
while getopts ":s" flag; do
  case "${flag}" in
    s) SKIPDATA="true" ;;
  esac
done

date=$(date +%y%m%d-%H%M%S)
postfix=$(openssl rand -hex 4)
logname="$date-$postfix"
export DATASET_DIR=$WORKDIR/dataset
export IMAGES_DIR=$DATASET_DIR/images
export CONFIG_DIR=$WORKDIR/configs-${logname}
export LOGDIR=$WORKDIR/logdir-${logname}
export SERVING_DIR=$WORKDIR/serving-${logname}

mkdir -p $WORKDIR
mkdir -p $DATASET_DIR
mkdir -p $IMAGES_DIR
mkdir -p $CONFIG_DIR
mkdir -p $LOGDIR
mkdir -p $SERVING_DIR

# ---- data preparation

if [[ -z "${SKIPDATA}" ]]; then
    catalyst-data process-images \
        --in-dir $DATADIR \
        --out-dir $IMAGES_DIR \
        --num-workers $NUM_WORKERS \
        --max-size $MAX_IMAGE_SIZE \
        --clear-exif

    catalyst-data tag2label \
        --in-dir $IMAGES_DIR \
        --out-dataset $DATASET_DIR/dataset_raw.csv \
        --out-labeling $DATASET_DIR/tag2class.json

    catalyst-data split-dataframe \
        --in-csv $DATASET_DIR/dataset_raw.csv \
        --tag2class $DATASET_DIR/tag2class.json \
        --tag-column=tag --class-column=class \
        --n-folds=5 --train-folds=0,1,2,3 \
        --out-csv=$DATASET_DIR/dataset.csv
fi


# ---- config preparation

python ./scripts/prepare_config.py \
    --in-template=$CONFIG_TEMPLATE \
    --out-config=$CONFIG_DIR/config.yml \
    --expdir=./src \
    --dataset-path=$DATASET_DIR \
    --num-workers=$NUM_WORKERS \
    --batch-size=$BATCH_SIZE \
    --max-image-size=$MAX_IMAGE_SIZE \
    --balance-strategy=$BALANCE_STRATEGY \
    --criterion=$CRITERION

cp -r ./configs/_common.yml $CONFIG_DIR/_common.yml


# ---- model training

catalyst-dl run \
    -C $CONFIG_DIR/_common.yml $CONFIG_DIR/config.yml \
    --logdir $LOGDIR $*


# ---- model tracing

catalyst-dl trace $LOGDIR -m $TRACED_METHOD --out-model $LOGDIR/traced.pth


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

cp $LOGDIR/traced.pth $SERVING_DIR/model.pth
cp $TAG2CLS_PATH $SERVING_DIR
