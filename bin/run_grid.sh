#!/usr/bin/env bash
set -e

if [[ -z "$RUN_CONFIG" ]]
then
      RUN_CONFIG=exp_splits.yml
fi

echo "Training...0"
catalyst-dl run \
    --config=./configs/classification/${RUN_CONFIG} \
    --logdir=-nodir --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str

echo "Training...1"
catalyst-dl run \
    --config=./configs/classification/${RUN_CONFIG} \
    --logdir=-nodir --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --model_params/embedding_net_params/hiddens=[128]:list

echo "Training...2"
catalyst-dl run \
    --config=./configs/classification/${RUN_CONFIG} \
    --logdir=-nodir --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --model_params/encoder_params/pooling=GlobalAvgPool2d:str \
    --model_params/embedding_net_params/hiddens=[256]:list

echo "Training...3"
catalyst-dl run \
    --config=./configs/classification/${RUN_CONFIG} \
    --logdir=-nodir --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --model_params/encoder_params/pooling=GlobalMaxPool2d:str \
    --model_params/embedding_net_params/hiddens=[256]:list

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${BASELOGDIR}
fi
