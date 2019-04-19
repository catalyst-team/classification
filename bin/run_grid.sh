#!/usr/bin/env bash
set -e

echo "Training...1"
catalyst-dl run \
    --config=./configs/finetune/exp_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose

echo "Training...2"
catalyst-dl run \
    --config=./configs/finetune/exp_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/encoder_params/pooling=GlobalAvgPool2d:str \
    --model_params/embedding_net_params/hiddens=[256]:list

echo "Training...3"
catalyst-dl run \
    --config=./configs/finetune/exp_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/encoder_params/pooling=GlobalMaxPool2d:str \
    --model_params/embedding_net_params/hiddens=[256]:list

echo "Training...4"
catalyst-dl run \
    --config=./configs/finetune/exp_splits.yml \
    --baselogdir=${BASELOGDIR} --verbose \
    --model_params/embedding_net_params/hiddens=[128]:list

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${BASELOGDIR}
fi
