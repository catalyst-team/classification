#!/usr/bin/env bash
set -e

if [[ -z "$RUN_CONFIG" ]]
then
      RUN_CONFIG=exp_splits.yml
fi

echo "training...and inference"
catalyst-dl run \
    --config=./configs/classification/${RUN_CONFIG} \
    --logdir="${LOGDIR}" \
    --out_dir="${LOGDIR}":str \
    --verbose

# docker trick
if [[ "$EUID" -eq 0 ]]; then
  chmod -R 777 ${LOGDIR}
fi
