#!/usr/bin/env bash
set -e

mkdir -p ${LOGDIR}/embeddings

echo "train embeddings creating..."
catalyst-data image2embedding \
    --in-csv=./data/dataset_train.csv \
    --img-col="filepath" \
    --img-datapath=./data/dataset/ \
    --out-npy=${LOGDIR}/embeddings/embeddings_train.npy \
    --arch="resnet18" \
    --pooling=GlobalMaxPool2d \
    --batch-size=64 \
    --n-workers=4 \
    --verbose

echo "valid embeddings creating..."
catalyst-data image2embedding \
    --in-csv=./data/dataset_valid.csv \
    --img-col="filepath" \
    --img-datapath=./data/dataset/ \
    --out-npy=${LOGDIR}/embeddings/embeddings_valid.npy \
    --arch="resnet18" \
    --pooling=GlobalMaxPool2d \
    --batch-size=64 \
    --n-workers=4 \
    --verbose

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${LOGDIR}
fi

