# Catalyst.Classification & Autolabel

***Intro: введение: в данном туториале вы сделаете кучу сложнейших всопроизводимых штук простым способом - конфигурируя конфиги. Каждый этап пару слов***



## 1.Classification




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


### 1.1 Install requirements

#### Using local environment: 
```pip install -r requirements.txt```

#### Using docker: 

This creates a build `catalyst-classification` with all needed libraries:
```bash
make classification
```

### 1.2 Get Dataset

![Ants-bees dataset example](/images/ant-bees.png "Ants-bees dataset example")

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

#### For your dataset
```bash
ln -s /path/to/your_dataset $(pwd)/data/dataset
```

### 1.3 Process the data
Using methods `tag2label` and `split-dataframe` dataset is prepearing to json like {“class_id”: class_column_from_dataset} and then spliting into train/valid folds. in this example one fold from five is using for validation, others for train.  

Here is description of another usefull metods for dataset preparation: `catalyst-data-documentation`[catalyst-data documentation](https://catalyst-team.github.io/catalyst/api/data.html)

#### For your dataset
To change num_classes in configs use:
```bash
export NUM_CLASSES=2; bash ./bin/prepare_configs.sh
```
#### In your local environment: 

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
#### Using docker:

```
docker run -it --rm -v $(pwd):/workspace/ catalyst-classification \
catalyst-data tag2label \
--in-dir=./data/dataset \
--out-dataset=./data/dataset_raw.csv \
--out-labeling=./data/tag2cls.json

docker run -it --rm -v $(pwd):/workspace/ catalyst-classification \
catalyst-data  split-dataframe  \
--in-csv=./data/dataset_raw.csv     \
--tag2class=./data/tag2cls.json     \
--tag-column=tag     \
--class-column=class     --n-folds=5     --train-folds=0,1,2,3    \
 --out-csv=./data/dataset.csv
```

### 1.4 Model training
Powerful configs allow us to investigate models in a controlled and reproducible way. We will perform the following experiments: 
- Softmax classification - 
- "Multilabel" classification -  
- "Multilabel" classification and FocalLoss - 
- Classification and rotation factor prediction - 

#### Config training

The config allows you to define:
- `data` path, batch size, num of workers and so on.
- `model_params` detailed configuration of models, including:
    - detailed architecture description
    - using or not pretrained model 
- `stages` you can configure training in several stages with different hyperparameters, optimizers and loss-functions. In our example:
     - first learn the head
     - then train the whole network  
     - produce predictions  
- augmentation parametrs
- technical parameters

#### Run in local environment: 

- Softmax classification:
```bash
catalyst-dl run --config=configs/exp_splits.yml
```

- "Multilabel" classification:
```bash
catalyst-dl run --config=configs/exp_splits_bce.yml
```

- "Multilabel" classification and FocalLoss:
```bash
catalyst-dl run --config=configs/exp_splits_focal.yml
```

- Classification and rotation factor prediction:
```bash
catalyst-dl run --config=configs/exp_splits_rotation.yml
```

#### Run in docker:
In docker run config of experiment is setting by `-e "RUN_CONFIG=exp_splits.yml"`:
```bash
export LOGDIR=$(pwd)/logs/classification
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   -e "RUN_CONFIG=exp_splits.yml"
   catalyst-classification bash bin/run_model.sh
```

#### Checkpoints

Checkpoins of all stages can be found in directory `./logs/classification/checkpoints`

At the end of each learning stage best checkpoints are logged:

- Stage 1:
```
19/20 * Epoch 19 (valid): _base/lr=0.0003 | _base/momentum=0.9000 | _timers/_fps=3749.7755 | _timers/batch_time=0.3367 | _timers/data_time=0.3266 | _timers/model_time=0.0099 | accuracy01=94.0848 | embeddings_loss=0.1721
Top best models:
logs/classification/checkpoints//stage1.11.pth  95.6473
logs/classification/checkpoints//stage1.14.pth  95.6473
logs/classification/checkpoints//stage1.7.pth   94.8661
```

- Stage 2:
```
9/10 * Epoch 30 (valid): _base/lr=0.0001 | _base/momentum=0.0000 | _timers/_fps=3681.9615 | _timers/batch_time=0.3884 | _timers/data_time=0.3788 | _timers/model_time=0.0093 | accuracy01=95.6473 | embeddings_loss=0.2165
Top best models:
logs/classification/checkpoints//stage2.21.pth  95.6473
logs/classification/checkpoints//stage2.27.pth  95.6473
logs/classification/checkpoints//stage2.30.pth  95.6473
```

#### Tensorboard metrics visualization 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
```
![Stage 1](/images/1_stage.jpg "Stage 1")


#### Index model preparation

```bash
export LOGDIR=$(pwd)/logs/classification
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR/embeddings/:/logdir/embeddings/ \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_index.sh
```


### 1.5 * TF.Projector and embeddings visualization

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

### 1.6 * Finding best start LR with LrFinder

```bash
export LOGDIR=$(pwd)/logs/lrfinder
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_lrfinder.sh
```

### 1.7 * Grid search visualization

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

## 2.Autolabel

***There will be introduction***

### Goals

Main
- tune ResnetEncoder
- train MultiHeadNet for image classification
- predict unlabelled dataset
- use most confident predictions as true labels
- repeat

### 2.1 Preparation

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


### 2.2 Model training

```bash
export GPUS=""
CUDA_VISIBLE_DEVICES="${GPUS}" bash ./bin/run_autolabel.sh \
    --data-raw ./data/data_raw/ \
    --data-clean ./data/data_clean/ \
    --baselogdir ./logs/autolabel \
    --n-trials 10 \
    --threshold 0.8
```
