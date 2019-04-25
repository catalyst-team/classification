#!/usr/bin/env bash
set -e

echo "projection creating..."
mkdir -p ${LOGDIR}/projector
catalyst-contrib project-embeddings \
   --in-npy=${LOGDIR}/embeddings/embeddings_train.npy \
   --in-csv="./data/dataset_train.csv" \
   --out-dir=${LOGDIR}/projector \
   --img-size=64 \
   --img-datapath=./data/dataset/ \
   --img-col="filepath" \
   --meta-cols="tag"

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${LOGDIR}
fi
