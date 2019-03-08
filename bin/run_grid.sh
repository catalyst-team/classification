#!/usr/bin/env bash
set -e

echo "Training...1"
catalyst-dl run \
    --expdir=finetune \
    --config=./configs/finetune/exp_splits.yml \
    --logdir=${BASELOGDIR} --verbose

echo "Training...2"
catalyst-dl run \
    --expdir=finetune \
    --config=./configs/finetune/exp_splits.yml \
    --logdir=${BASELOGDIR} --verbose \
    --model_params/encoder_params/pooling=GlobalAvgPool2d:str \
    --model_params/head_params/hiddens=[512]:list

echo "Training...3"
catalyst-dl run \
    --expdir=finetune \
    --config=./configs/finetune/exp_splits.yml \
    --logdir=${BASELOGDIR} --verbose \
    --model_params/encoder_params/pooling=GlobalMaxPool2d:str \
    --model_params/head_params/hiddens=[512]:list

echo "Training...4"
catalyst-dl run \
    --expdir=finetune \
    --config=./configs/finetune/exp_splits.yml \
    --logdir=${BASELOGDIR} --verbose \
    --model_params/head_params/emb_size=128:int

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${BASELOGDIR}
fi
