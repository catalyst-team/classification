#!/usr/bin/env bash
set -e

# --- data

#mkdir -p ./data
#wget https://www.dropbox.com/s/8aiufmo0yyq3cf3/ants_bees_cleared_190806.tar.gz
#tar -xf ants_bees_cleared_190806.tar.gz &>/dev/null
#mv ants_bees_cleared_190806 ./data/origin

# ---- environment variables

export CONFIG_TEMPLATE=./configs/templates/class.yml
export DATADIR=./data/origin
export NUM_WORKERS=4
export BATCH_SIZE=64
export MAX_IMAGE_SIZE=224

SKIPDATA=""
while getopts ":s" flag; do
  case "${flag}" in
    s) SKIPDATA="true" ;;
  esac
done

if [[ -z "$NUM_WORKERS" ]]; then
      NUM_WORKERS=4
fi

if [[ -z "$MAX_IMAGE_SIZE" ]]; then
      MAX_IMAGE_SIZE=224
fi

date=$(date +%y%m%d-%H%M%S-%3N)
export WORKDIR=./logs
export DATASET_DIR=$WORKDIR/dataset
export IMAGES_DIR=$DATASET_DIR/images
export CONFIG_DIR=$WORKDIR/configs-${date}
export LOGDIR=$WORKDIR/logdir-${date}
export SERVING_DIR=$WORKDIR/serving-${date}

mkdir -p $WORKDIR
mkdir -p $DATASET_DIR
mkdir -p $IMAGES_DIR
mkdir -p $CONFIG_DIR
mkdir -p $LOGDIR
mkdir -p $SERVING_DIR

# ---- data preparation

if [[ -z "${SKIPDATA}" ]]; then
    catalyst-data process-images \
        --in-dir $DATADIR \
        --out-dir $IMAGES_DIR \
        --num-workers $NUM_WORKERS \
        --max-size $MAX_IMAGE_SIZE \
        --clear-exif

    catalyst-data tag2label \
        --in-dir $IMAGES_DIR \
        --out-dataset $DATASET_DIR/dataset_raw.csv \
        --out-labeling $DATASET_DIR/tag2class.json

    catalyst-data split-dataframe \
        --in-csv $DATASET_DIR/dataset_raw.csv \
        --tag2class $DATASET_DIR/tag2class.json \
        --tag-column=tag --class-column=class \
        --n-folds=5 --train-folds=0,1,2,3 \
        --out-csv=$DATASET_DIR/dataset.csv
fi

# ---- config preparation

python ./scripts/prepare_config.py \
    --in-template=$CONFIG_TEMPLATE \
    --out-config=$CONFIG_DIR/config.yml \
    --expdir=./src \
    --dataset-path=$DATASET_DIR \
    --max-image-size=$MAX_IMAGE_SIZE \
    --num-workers=$NUM_WORKERS \
    --batch-size=$BATCH_SIZE

cp -r ./configs/_common.yml $CONFIG_DIR/_common.yml


# ---- model training

catalyst-dl run \
    -C $CONFIG_DIR/config.yml $CONFIG_DIR/_common.yml \
    --logdir $LOGDIR --check

# ---- model tracing

catalyst-dl trace $LOGDIR -m forward_embeddings

# ---- model serving

TAG2CLS_PATH=$(python << EOF
import json
with open("${LOGDIR}/configs/_config.json") as f:
    conf = json.load(f)
    path = conf["stages"]["data_params"]["datapath"].replace("images", "") \
        + "tag2class.json"
    print(path)
EOF
)

cp $LOGDIR/traced.pth $SERVING_DIR/model.pth
cp TAG2CLS_PATH $SERVING_DIR
