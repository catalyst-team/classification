[![Build Status](https://travis-ci.com/catalyst-team/classification.svg?branch=master)](https://travis-ci.com/catalyst-team/classification)
[![Telegram](./pics/telegram.svg)](https://t.me/catalyst_team)
[![Gitter](https://badges.gitter.im/catalyst-team/community.svg)](https://gitter.im/catalyst-team/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](./pics/slack.svg)](https://opendatascience.slack.com/messages/CGK4KQBHD)
[![Donate](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/patreon.png)](https://www.patreon.com/catalyst_team)

# Catalyst.Classification

You will learn how to build image classification pipeline with transfer learning using the Catalyst framework.

### Goals
- Install requirements
- Get Dataset
- Run classification pipeline: raw data → production-ready model

### 1.1 Install requirements

#### Using local environment: 

```bash
pip install -r requirements/requirements_min.txt
```

#### Using docker:

This creates a build `catalyst-classification` with the necessary libraries:
```bash
make docker-build
```

### 1.2 Get Dataset

```bash
mkdir data
```
You can use one of the following datasets

1. Ant and Bees
```bash
    wget https://www.dropbox.com/s/8aiufmo0yyq3cf3/ants_bees_cleared_190806.tar.gz
    tar -xf ants_bees_cleared_190806.tar.gz &>/dev/null
    mv ants_bees_cleared_190806 ./data/origin
 ```
2. Flowers
```bash
    # https://www.kaggle.com/alxmamaev/flowers-recognition
    wget https://www.dropbox.com/s/lwcvy4eb68drvs3/flowers.tar.gz
    tar -xf flowers.tar.gz &>/dev/null
    mv flowers ./data/origin
 ```

3. Artworks:
 ```bash
    # https://www.kaggle.com/ikarus777/best-artworks-of-all-time
    wget https://www.dropbox.com/s/ln4ot1fu2sgtgvg/artworks.tar.gz
    tar -xf artworks.tar.gz &>/dev/null
    mv artworks ./data/origin
```

#### For your dataset

Make sure, that final folder structure with training data:
```bash
/path/to/your_dataset/
        class_name_1/
            images
        class_name_2/
            images
        ...
        class_name_100500/
            ...
```
The easiest way is to move your data:
```bash
mv /path/to/your_dataset/* /catalyst.classification/data/origin 
``` 
In that way you can run pipeline with defaul settings. 

##### If you still leave data in `/path/to/your_dataset/` 
##### In local environment:

```bash
ln -s /path/to/your_dataset $(pwd)/data/origin
```
Or just set path to your dataset `DATADIR=/path/to/your_dataset` when you start the pipeline.

##### Using docker

You need to set:

```bash
   -v /path/to/your_dataset:/data \ #instead default  $(pwd)/data/origin:/data
 ```
 in the script below to start the pipeline.

### 1.3 Classification pipeline
#### Fast&Furious: raw data → production-ready model

The pipeline will automatically guide you from raw data to the production-ready model. 

All you need is to make a couple of settings.
##### Run in local environment: 

```bash	
CUDA_VISIBLE_DEVICES=0 \	
CUDNN_BENCHMARK="True" \	
CUDNN_DETERMINISTIC="True" \	
WORKDIR=./logs \	
DATADIR=./data/origin \	
MAX_IMAGE_SIZE=224 \  # 224 or 448 works good	
BALANCE_STRATEGY=256 \  # images in epoch per class, 1024 works good	
CONFIG_TEMPLATE=./configs/templates/ce.yml \	
NUM_WORKERS=4 \	
BATCH_SIZE=256 \	
bash ./bin/catalyst-classification-pipeline.sh	
```
You can use [W&B](https://www.wandb.com/) account for viaualisation:

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```

##### Run in docker:

```bash
export LOGDIR=$(pwd)/logs
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/  $(pwd)/data/origin:/data \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "USE_WANDB=1" \
   -e "LOGDIR=/logdir" \
   -e "CUDNN_BENCHMARK='True'" \	
   -e "CUDNN_DETERMINISTIC='True'" \	
   -e "WORKDIR=/logdir" \	
   -e "DATADIR=/data" \	
   -e "MAX_IMAGE_SIZE=224" \  	
   -e "BALANCE_STRATEGY=256" \ 	
   -e "CONFIG_TEMPLATE=./configs/templates/ce.yml" \	
   -e "NUM_WORKERS=4" \	
   -e "BATCH_SIZE=256" \	
   catalyst-classification ./bin/catalyst-classification-pipeline.sh
```

During the pipeline model will be trained sequentially in two stages, also in the first stage we will train several heads simultaneously.
Experiments can be performed using pre-trained model ResNet-18 with with the following configs:
- `ce.yml`  using `CrossEntropyLoss`
- `bce.yml` using `BCEWithLogits` Loss
- `focal.yml` using `FocalLossMultiClass` Loss


### Results
Configs of all experiments, configurations of data for experiments, logs and checkpoins can be found in directory `./logs`
