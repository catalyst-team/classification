#!/usr/bin/env bash
set -e

echo "training...and inference"
catalyst-dl run \
    --expdir=src \
    --config=configs/exp_splits.yml \
    --logdir=${LOGDIR} \
    --verbose

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
