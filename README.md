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
- Process the data with Catalyst.Data
- FineTune ResnetEncoder with Catalyst Config API
- Train ResnetEncoder with different loss functions for image classification

### 1.1 Install requirements

#### Using local environment: 
```bash
pip install -r requirements/requirements_min.txt
```

#### Using docker:

This creates a build `catalyst-classification` with all needed libraries:
```bash
make classification
```

### 1.2 Get Dataset
![MNIST dataset example](/images/dataset_sample.png "Mnist dataset example")

Get the [dataset example](https://www.dropbox.com/s/eeme52kwnvz255d/mnist.tar.gz) and unpack it to `data` folder:
```bash
wget -P ./data/ https://www.dropbox.com/s/wviqz4g55kl4zft/trainingSet.tar.gz
tar -xvf ./data/trainingSet.tar.gz -C ./data
mv ./data/trainingSet ./data/dataset
```
Or get the full MNIST dataset (4200 jpgs) from [kaggle competition.](https://www.kaggle.com/scolianni/mnistasjpg) 
```bash
wget -P ./data/ https://www.dropbox.com/s/eeme52kwnvz255d/mnist.tar.gz
tar -xvf ./data/mnist.tar.gz -C ./data
mv ./data/mnist ./data/dataset
```
Final folder structure with training data:
```bash
catalyst.classification/data/
    dataset/
        0/
            ...
        1/
            ...
        ...
        9/
            ...
```

#### For your dataset
```bash
ln -s /path/to/your_dataset $(pwd)/data/dataset
```

### 1.3 Process the data
Using methods `tag2label` and `split-dataframe` dataset is prepearing to json like 
```json
{"class_name": class_num}
``` 
and then spliting into train/valid folds. in this example one fold from five is using for validation, others for train.

Here is description of another usefull metods for dataset preparation: [catalyst-data documentation](https://catalyst-team.github.io/catalyst/api/data.html)

#### For your dataset
To change num_classes in configs use:
```bash
export NUM_CLASSES=10
./bin/prepare_configs.sh
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

Quite rarely we have a dataset for learning the deep convolutional network from scratch. Usually we have a small dataset and use the weights obtained from model with the same architecture have trained on a large dataset, such as ImageNet, containing more than a million images.

Two basic scenarios of transfer training will be used:

1. We initialize the network with a pre-trained network, for example, an Imagenet-trained dataset. We freeze weights for the entire network, with the exception the last fully connected layers, called "head". We initialize head's weights with random, and only this one is trained.
2. We initialize the network with a pre-trained network, for example, an Imagenet trained dataset. Further we train a network entirely on ours dataset.

We will use approaches sequentially in two stages, also in the first stage we will train several heads simultaneously.
We will perform the following experiments using pre-trained model ResNet-18:
- Two stages traininig classification using `Softmax`
- Two stages training "Multilabel" classification using `BCEWithLogitsLoss`
- Two stages training "Multilabel" classification using `FocalLossMultiClass`

#### Config training
The config allows you to define:
- `data_params` path, batch size, num of workers and so on.
- `model_params` detailed configuration of models, including:
    - detailed architecture description
    - using or not pretrained model
- `stages` you can configure training or inference in several stages with different hyperparameters, optimizers, callbacks and loss-functions. In our example:
     - using pretrained model
     - first learn the head(s)
     - then train the whole network
     - produce predictions
- augmentation parametrs
- other experiment hyperparameters

#### Run in local environment: 
- Classification using `Softmax`:
```bash
catalyst-dl run --config=configs/exp_splits.yml --verbose
```

- "Multilabel" classification using `BCEWithLogitsLoss`:
```bash
catalyst-dl run --config=configs/exp_splits_bce.yml --verbose
```

-  "Multilabel" classification using `FocalLossMultiClass`:
```bash
catalyst-dl run --config=configs/exp_splits_focal.yml --verbose
```

- Two stages trained MultiHeadNet classification using `CrossEntropyLoss` and with augmentations prediction:
```bash
catalyst-dl run --config=configs/exp_splits_augs.yml --verbose
```

To use WandbRunner instead of usual runner add `USE_WANDB=1`, like
```bash
USE_WANDB=1 catalyst-dl run --config=<you choose> --verbose
```

#### Run in docker:
In docker run config of experiment is setting by `-e "RUN_CONFIG=exp_splits.yml"`:
```bash
export LOGDIR=$(pwd)/logs/classification
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "USE_WANDB=1" \
   -e "LOGDIR=/logdir" \
   -e "RUN_CONFIG=exp_splits.yml"
   catalyst-classification bash bin/run_model.sh
```

#### Checkpoints
Checkpoins of all stages can be found in directory `./logs/classification/checkpoints`

At the end of each learning stage best checkpoints are logged:

- Stage 1:
```bash
19/20 * Epoch 19 (valid): _base/lr=0.0003 | _base/momentum=0.9000 | _timers/_fps=6189.0000 | _timers/batch_time=0.2606 | _timers/data_time=0.2549 | _timers/model_time=0.0056 | accuracy01=56.4732 | embeddings_loss=1.3530
Top best models:
logs/classification/checkpoints//stage1.17.pth	58.2589
logs/classification/checkpoints//stage1.19.pth	56.4732
logs/classification/checkpoints//stage1.7.pth	55.3571
```
- Stage 2:
```bash
9/10 * Epoch 30 (valid): _base/lr=0.0001 | _base/momentum=0.0000 | _timers/_fps=4985.2862 | _timers/batch_time=0.2771 | _timers/data_time=0.2703 | _timers/model_time=0.0067 | accuracy01=53.2366 | embeddings_loss=1.4358
Top best models:
logs/classification/checkpoints//stage2.23.pth	60.4911
logs/classification/checkpoints//stage2.26.pth	56.6964
logs/classification/checkpoints//stage2.28.pth	54.6875
```
