# Catalyst.Classification & Autolabel

Framework provides powerful configs allow to optimize configuration of the whole pipeline of classification in a controlled and reproducible way.

The framework also provide tools to:
 - create KNN index model
 - create and visualize embeddings
 - find best starting learning rate
 - plot grid search metrics and compare different approaches
 - perform autolabel

## 1. Classification

You will learn how to build image classification pipeline with transfer learning using the "Catalyst" framework.

### Goals
- FineTune ResnetEncoder
- Train ResnetEncoder with different loss functions for image classification
- Train MultiHeadNet for "multilabel" image classification with augmentations prediction
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
- Two stages training classification using `CrossEntropyLoss` and with augmentations prediction

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

<img src="/images/classification_metrics.png" title="classificaton" align="left">

Confusion matrices:

<img src="/images/cm_classes.png" width="350" title="labels confusion matrix" align="left">
<img src="/images/cm_rotation.png" width="350" title="rotation confusion matrix" align="middle">


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

#### Create KNN Index model

Factor-based model PCA reduce dimenshionality of embeddins space, then KNN model is used to calculate the score distance in score space and transform data into a fast indexing structure. 

As a result, we have index model which allow as to implement fast similarity search on a large number of high-dimensional vectors. KNN Index model also allow fast and easy handle out-of-class predictions.

```bash
export LOGDIR=$(pwd)/logs/classification
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR:/logdir \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_index.sh
```
Out:
```
index model creating...
[==       Loading features       ==]
[==     Transforming features    ==]
[ Explained variance ratio: 0.9602 ]
[==        Saving pipeline       ==]
[==  Adding features to indexer  ==]
[==        Creating index        ==]
[==         Saving index         ==]
index model testing...
[==       Loading features       ==]
[==        Loading index         ==]
[==      Recall@ 1: 97.44%      ==]
[==      Recall@ 3: 98.72%      ==]
[==      Recall@ 5: 100.0%      ==]
[==      Recall@10: 100.0%      ==]
```

The result of the work are the following files:
```bash
${LOGDIR}/knn.embeddings.bin
${LOGDIR}/predictions/infer.embeddings.pca.npy 
${LOGDIR}/pipeline.embeddings.pkl 
```

### 1.5 TF.Projector and embeddings visualization

#### Embeddings creation

Embeds images from dataset with specified neural net architecture
```bash
export LOGDIR=$(pwd)/logs/projector
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR/embeddings/:/logdir/embeddings/ \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_embeddings.sh
```

#### Embeddings projection

```bash
export LOGDIR=$(pwd)/logs/projector
docker run -it --rm --shm-size 8G \
   -v $(pwd):/workspace/ \
   -v $LOGDIR:/logdir  \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_projector.sh
```

#### Embeddings visualization 

```bash
export LOGDIR=$(pwd)/logs/projector
tensorboard --logdir=$LOGDIR/projector
```
<img src="/images/projector_2d.png" width="500" title="projector">

### 1.6 Finding best start LR with LrFinder
Put trainig with callback LRFinder to find the optimal learning rate range for your model and dataset.
In the kofig there are:
- the scale in which the Learning rate increases:  linear or log 
- final learning rate
- num steps

```bash
export LOGDIR=$(pwd)/logs/lrfinder
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "LOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_lrfinder.sh
```
#### Tensorboard metrics visualization  

```bash
export LOGDIR=$(pwd)/logs
tensorboard --logdir=$LOGDIR
```
![LRFinder](/images/LRFinder.png "LrFinder")

### 1.7 Grid search visualization

#### Hyperparameters grid search training
Specifying parameters of trainings including hyperparametres and model parametres you can by one file `./bin/run_grid.sh` configurate and run the sequence of experiments with logging.

```bash
export BASELOGDIR=$(pwd)/logs/grid
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "BASELOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_grid.sh
```

#### Tensorboard metrics visualization 
```bash
tensorboard --logdir=./logs
```
Classification metrics during learning:

![grid search](/images/grid_search.png "grid search")

Configs of experiments can be found in directory `./logs/grid/`


#### KFold training
You can by one file `./bin/run_kfold.sh` configurate and run the sequence of kfold trainings.

```bash
export BASELOGDIR=$(pwd)/logs/classification/kfold
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $BASELOGDIR:/logdir/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "BASELOGDIR=/logdir" \
   catalyst-classification bash ./bin/run_kfold.sh
```
#### Tensorboard metrics visualization 
```bash
tensorboard --logdir=./logs
```
Classification metrics during learning:

![KFold training](/images/kfold.png "KFold training")

## 2. Autolabel

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
        all/
            ...
    data_clean/
        0/
            ...
        1/
            ...
```

### 2.2 Model training

```bash
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ \
   -e "CUDA_VISIBLE_DEVICES=0" \
      catalyst-classification bash ./bin/run_autolabel.sh \
    --data-raw ./data/data_raw/ \
    --data-clean ./data/data_clean/ \
    --baselogdir ./logs/autolabel \
    --n-trials 10 \
    --threshold 0.8
```

### 2.3 Results of autolabeling
Out:
```
Predicted: 23 (100.00%)
...
Pseudo Lgabeling done. Nothing more to label.
```
Logs for trainings visualisation can be found here: `./logs/autolabel` 

Labeled raw data can be found here: `/data/data_clean/dataset.csv`
