# Catalyst.Classification & Autolabel

***Intro: введение: в данном туториале вы сделаете кучу сложнейших всопроизводимых штук простым способом - конфигурируя конфиги. Каждый этап пару слов***

## Classification

### Goals

Main
- tune ResnetEncoder - ***t***
- train MultiHeadNet for image classification - ***t***
- learn embeddings representation - ***t***
- create knn index model - ***t***
- or train MultiHeadNet for "multilabel" image classification - ***t***

Additional
- visualize embeddings with TF.Projector - ***t***
- find best starting lr with LRFinder - ***t***
- plot grid search metrics and compare different approaches - ***t***

### Preparation

#### 1. Get Dataset

![Ants-bees dataset example](/images/ant-bees-example.PNG "Ants-bees dataset example")

Get the [data](https://www.dropbox.com/s/9438wx9ku9ke1pt/ants_bees.tar.gz) and unpack it to `data` folder:
```bash
wget -P ./data/ https://www.dropbox.com/s/9438wx9ku9ke1pt/ants_bees.tar.gz
tar -xvf ./data/ants_bees.tar.gz -C ./data
mv ./data/ants_bees ./data/dataset

```

Final folder structure with training data:
```bash
catalyst.classification/data/
    dataset/
        ants/
            ...
        bees/
            ...
```

For your dataset use:
```bash
ln -s /path/to/your_dataset $(pwd)/data/dataset
```


#### 2. Install requirements

if you want use your local environment: 
```pip install -r requirements.txt```

!add jpeg4py!

if you want use docker: 

```bash
make classification
```
or 
```
docker build -t catalyst-classification:latest . -f docker/Dockerfile
```
#### 3. Process the data
In your local environment: 

```bash
catalyst-data tag2label \
    --in-dir=./data/dataset \
    --out-dataset=./data/dataset_raw.csv \
    --out-labeling=./data/tag2cls.json

catalyst-data split-dataframe \
    --in-csv=./data/dataset_raw.csv \
    --tag2class=./data/tag2cls.json \
    --tag-column=tag \
    --class-column=class \
    --n-folds=5 \
    --train-folds=0,1,2,3 \
    --out-csv=./data/dataset.csv
```
In docker:
```
docker run -it --rm -v $(pwd):/workspace/ catalyst-classification catalyst-data tag2label  --in-dir=./data/dataset     --out-dataset=./data/dataset_raw.csv     --out-labeling=./data/tag2cls.json
```
```
docker run -it --rm -v $(pwd):/workspace/ catalyst-classification catalyst-data  split-dataframe  \
    --in-csv=./data/dataset_raw.csv     \
--tag2class=./data/tag2cls.json     \
--tag-column=tag     \
--class-column=class     --n-folds=5     --train-folds=0,1,2,3    \
 --out-csv=./data/dataset.csv
```

To change num_classes in configs use:
```bash
export NUM_CLASSES=2; bash ./bin/prepare_configs.sh
```

This creates a build `catalyst-classification` with all needed libraries.

### Model training

Local run with softmax classification:
```bash
catalyst-dl run --config=configs/exp_splits.yml
```

Local run with "multilabel" classification:
```bash
catalyst-dl run --config=configs/exp_splits_bce.yml
```

Local run with "multilabel" classification and FocalLoss:
```bash
catalyst-dl run --config=configs/exp_splits_focal.yml
```

Local run with classification and rotation factor prediction:
```bash
catalyst-dl run --config=configs/exp_splits_rotation.yml
```

Docker run:
```bash
export LOGDIR=$(pwd)/logs/classification
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash bin/run_model.sh
```

#### Tensorboard metrics visualization 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
```


#### Index model preparation

```bash
export LOGDIR=$(pwd)/logs/classification
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR/embeddings/:/logdir/embeddings/ \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_index.sh
```


## * TF.Projector and embeddings visualization

#### Embeddings creation

```bash
export LOGDIR=$(pwd)/logs/projector
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR/embeddings/:/logdir/embeddings/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_embeddings.sh
```

#### Embeddings projection

```bash
export LOGDIR=$(pwd)/logs/projector
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR/embeddings/:/logdir/embeddings/ \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_projector.sh
```

#### Embeddings visualization 

```bash
export LOGDIR=$(pwd)/logs/projector
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=$LOGDIR/projector
```

## * Finding best start LR with LrFinder

```bash
export LOGDIR=$(pwd)/logs/lrfinder
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_lrfinder.sh
```

## * Grid search visualization

#### Hyperparameters grid search training

```bash
export BASELOGDIR=$(pwd)/logs/grid
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "BASELOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_grid.sh
```


#### KFold training

```bash
export BASELOGDIR=$(pwd)/logs/classification/kfold
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "BASELOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_kfold.sh
```

## Autolabel

### Goals

Main
- tune ResnetEncoder
- train MultiHeadNet for image classification
- predict unlabelled dataset
- use most confident predictions as true labels
- repeat

### Preparation

```bash
catalyst.classification/data/
    data_raw/
        raw/
            ...
    data_clean/
        ants/
            ...
        bees/
            ...
```


### Model training

```bash
export GPUS=""
CUDA_VISIBLE_DEVICES="${GPUS}" bash ./bin/run_autolabel.sh \
    --data-raw ./data/data_raw/ \
    --data-clean ./data/data_clean/ \
    --baselogdir ./logs/autolabel \
    --n-trials 10 \
    --threshold 0.8
```
