#!/usr/bin/env bash

set -e

prject_root=$(realpath $(dirname $0)/../../../)

logdir=$(realpath $1)

cd $prject_root

if [[ -z $logdir ]]; then
    echo "select logdir as first arg"
    exit
fi

catalyst-dl trace $logdir -m predict_embedding

tag_to_class_path=$(python << EOF
import json
with open("${logdir}/configs/_config.json") as f:
    conf = json.load(f)
    path = conf["stages"]["data_params"]["datapath"].replace("images", "") \
        + "tag2class.json"
    print(path)
EOF
)

service=classifier
date=$(date +%y%m%d)

if [[ ! -z ${COMMENT} ]]; then
    COMMENT="-${COMMENT}"
fi

model_root=${MODEL_ROOT:-$prject_root/checkpoints}

model_name=${service}${COMMENT}-${date}
model_path=$model_root/$model_name
mkdir -p $model_path
cp $logdir/traced.pth $model_path/model.pth
cp $tag_to_class_path $model_path

echo
echo "Created new model named $model_name"
