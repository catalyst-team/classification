#!/usr/bin/env bash
set -e

if [[ -z "$RUN_CONFIG" ]]
then
      RUN_CONFIG=exp_splits.yml
fi

echo "Training...0"
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=none --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --stages/data_params/in_csv=./data/dataset.csv:str \
    --stages/data_params/in_csv_train=None:str \
    --stages/data_params/in_csv_valid=None:str \
    --stages/data_params/train_folds=[1,2,3,4]:list

echo "Training...1"
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=none --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --stages/data_params/in_csv=./data/dataset.csv:str \
    --stages/data_params/in_csv_train=None:str \
    --stages/data_params/in_csv_valid=None:str \
    --stages/data_params/train_folds=[0,2,3,4]:list

echo "Training...2"
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=none --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --stages/data_params/in_csv=./data/dataset.csv:str \
    --stages/data_params/in_csv_train=None:str \
    --stages/data_params/in_csv_valid=None:str \
    --stages/data_params/train_folds=[0,1,3,4]:list

echo "Training...3"
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=none --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --stages/data_params/in_csv=./data/dataset.csv:str \
    --stages/data_params/in_csv_train=None:str \
    --stages/data_params/in_csv_valid=None:str \
    --stages/data_params/train_folds=[0,1,2,4]:list

echo "Training...4"
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=none --baselogdir=${BASELOGDIR} --verbose \
    --stages/infer=None:str \
    --stages/data_params/in_csv=./data/dataset.csv:str \
    --stages/data_params/in_csv_train=None:str \
    --stages/data_params/in_csv_valid=None:str \
    --stages/data_params/train_folds=[0,1,2,3]:list

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${BASELOGDIR}
fi
