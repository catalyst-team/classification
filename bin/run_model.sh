#!/usr/bin/env bash
set -e

echo "training...and inference"
catalyst-dl run \
    --config=configs/finetune/exp_splits.yml \
    --logdir="${LOGDIR}" \
    --verbose

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${LOGDIR}
fi
