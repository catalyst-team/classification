#!/usr/bin/env bash

set -e

usage()
{
  cat << USAGE >&2
Usage: $(basename "$0") [OPTION...]

  --no-process                      Skip data pre-processing
  --no-split                        Skip data splitting into folds
  --max-image-size MAX_IMAGE_SIZE   Target size of images e.g. 224, 448
  -j, --num-workers NUM_WORKERS     Number of data loading/processing workers
  --datadir DATADIR                 Path to root directory with original images
  --workdir WORKDIR                 Working directory, where to store the result
  --dataset-dir DATASET_DIR         Path to root directory with dataset info, e.g. fold splits, results of labeling
  --images-dir IMAGES_DIR           Path to root directory where to store images via script processing

Example:
  ./bin/_data_preparation.sh \\
    --max-image-size 224 \\
    --num-workers 4 \\
    --datadir ./data/origin \\
    --workdir ./logs \\
    --dataset-dir ./logs/dataset \\
    --images-dir ./logs/images
USAGE
  exit 1
}


# ---- environment variables

NUM_WORKERS=${NUM_WORKERS:=4}
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:=224}
DATADIR=${DATADIR:="./data/origin"}
WORKDIR=${WORKDIR:="./logs"}
DATASET_DIR=${DATASET_DIR:=${WORKDIR}/dataset}
IMAGES_DIR=${IMAGES_DIR:=${DATASET_DIR}/images}

PROCESS_DATA="true"
SPLIT_DATASET="true"

while (( "$#" )); do
  case "$1" in
    --no-process)
      PROCESS_DATA="false"
      shift 1
      ;;
    --no-split)
      SPLIT_DATASET="false"
      shift 1
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE=$2
      shift 2
      ;;
    -j|--num-workers)
      NUM_WORKERS=$2
      shift 2
      ;;
    --datadir)
      DATADIR=$2
      shift 2
      ;;
    --workdir)
      WORKDIR=$2
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR=$2
      shift 2
      ;;
    --images-dir)
      IMAGES_DIR=$2
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done


# ---- data preparation

if [[ "${PROCESS_DATA}" == "true" ]]; then
  mkdir -p ${IMAGES_DIR}

  catalyst-data process-images \
    --in-dir ${DATADIR} \
    --out-dir ${IMAGES_DIR} \
    --num-workers ${NUM_WORKERS} \
    --max-size ${MAX_IMAGE_SIZE} \
    --clear-exif
fi

if [[ "${SPLIT_DATASET}" == "true" ]]; then
  catalyst-data tag2label \
    --in-dir ${IMAGES_DIR} \
    --out-dataset ${DATASET_DIR}/dataset_raw.csv \
    --out-labeling ${DATASET_DIR}/tag2class.json

  catalyst-data split-dataframe \
    --in-csv ${DATASET_DIR}/dataset_raw.csv \
    --tag2class ${DATASET_DIR}/tag2class.json \
    --tag-column=tag --class-column=class \
    --n-folds=5 --train-folds=0,1,2,3 \
    --out-csv=${DATASET_DIR}/dataset.csv
else
  cp ${DATADIR}/*.{csv,json} ${DATASET_DIR}/ 2>/dev/null || :
fi
