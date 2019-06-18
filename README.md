# Catalyst.Classification & Autolabel

Framework provides powerful configs allow to optimize configuration of the whole pipeline of classification in a controlled and reproducible way. 

The also framework provide tools to:
 - create KNN index model and visualize embeddings
 - find best starting learning rate
 - plot grid search metrics and compare different approaches
 - perform autolabel 
 
## 1.Classification

You will learn how to build image classification pipeline with transfer learning using the "Catalyst" framework. 

### Goals
- FineTune ResnetEncoder.
- FineTune MultiHeadNet for image classification
- FineTune MultiHeadNet for "multilabel" image classification
- Learn embeddings representation
- Create KNN index model 
- Visualize embeddings with TF.Projector 
- Find best starting lr with LRFinder 
- Plot grid search metrics and compare different approaches 

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
```bash
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

Quite rarely is there a dataset for learning the deep convolutional network from scratch. Usually we have a small dataset and use the weights obtained when training model with the same architecture on a large dataset, such as ImageNet, containing more than a million images.

Two basic scenarios of transfer training will be used:

1. We initialize the network with a pre-trained network, for example, an Imagenet-trained dataset. We freeze weights for the entire network, with the exception the last fully connected layers, called "head". We initialize head's weights with random, and only this one is trained.
2. We initialize the network with a pre-trained network, for example, an Imagenet trained dataset. Further we train a network entirely on ours dataset.

We will use approaches sequentially in two stages, also in the first stage we will train several heads simultaneously.
We will perform the following experiments using pre-trained model ResNet-18: 
- Two stages traininig classification using `Softmax` 
- Two stages training "Multilabel" classification using `BCEWithLogitsLoss`
- Two stages training MultiHeadNet "Multilabel" classification using `FocalLossMultiClass`  and with rotation factor prediction 
- Two stages training MultiHeadNet classification using `CrossEntropyLoss` and with rotation factor prediction 

#### Config training
The config allows you to define:
- `data` path, batch size, num of workers and so on.
- `model_params` detailed configuration of models, including:
    - detailed architecture description
    - using or not pretrained model 
- `stages` you can configure training in several stages with different hyperparameters, optimizers, callbacks and loss-functions. In our example:
     - first learn the head(s)
     - then train the whole network  
     - produce predictions  
- augmentation parametrs
- technical parameters

#### Run in local environment: 
- Classification using `Softmax`:
```bash
catalyst-dl run --config=configs/exp_splits.yml
```

- "Multilabel" classification using `BCEWithLogitsLoss`:
```bash
catalyst-dl run --config=configs/exp_splits_bce.yml
```

-  MultiHeadNet "Multilabel" classification using `FocalLossMultiClass`  and with rotation factor predictions:
```bash
catalyst-dl run --config=configs/exp_splits_focal.yml
```

- Two stages trained MultiHeadNet classification using `CrossEntropyLoss` and with rotation factor prediction :
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

#### Tensorboard metrics visualization 
```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
```
Classification metrics during learning:
![classificaton](/images/classification.png "classificaton")

Confusion matrices:
![labels confusion matrix](/images/cm_classes.png "labels confusion matrix")

![rotation confusion matrix](/images/cm_rotation.png "rotation confusion matrix")

#### Checkpoints
Checkpoins of all stages can be found in directory `./logs/classification/checkpoints`

At the end of each learning stage best checkpoints are logged:

- Stage 1:
```bash
19/20 * Epoch 19 (valid): _base/lr=0.0003 | _base/momentum=0.9000 | _timers/_fps=42.7747 | _timers/batch_time=2.7741 | _timers/data_time=0.2625 | _timers/model_time=2.5115 | accuracy01=92.0759 | auc/_mean=0.9793 | auc/class_0=0.9789 | auc/class_1=0.9796 | embeddings_loss=0.2581
Top best models:
/logdir/checkpoints//stage1.18.pth  97.6562
/logdir/checkpoints//stage1.16.pth  96.4286
/logdir/checkpoints//stage1.7.pth  95.6473
```

- Stage 2:
```bash
9/10 * Epoch 30 (valid): _base/lr=0.0001 | _base/momentum=0.0000 | _timers/_fps=37.3813 | _timers/batch_time=17.8423 | _timers/data_time=15.5075 | _timers/model_time=2.3346 | accuracy01=90.5134 | auc/_mean=0.9783 | auc/class_0=0.9776 | auc/class_1=0.9789 | embeddings_loss=0.2753
Top best models:
/logdir/checkpoints//stage2.23.pth  99.2188
/logdir/checkpoints//stage2.22.pth  98.6473
/logdir/checkpoints//stage2.29.pth  98.6473
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


### 1.5 TF.Projector and embeddings visualization

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

### 1.6 Finding best start LR with LrFinder

```bash
export LOGDIR=$(pwd)/logs/lrfinder
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_lrfinder.sh
```

### 1.7 Grid search visualization

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

### Goals

The classical way to reduce the amount of unlabeled data by having a trained model would be to run unlabeled dataset through the model and automatically label images with confidence of label prediction above the threshold. Then automatically labeled data pushing in the training process so as to optimize prediction accuracy.

To run the iteration process we need to specify number of iterations `n-trials` and `threshold` of confidence to label image.

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
