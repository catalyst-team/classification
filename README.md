<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL & RL!**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)
[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-twitter-499feb)](https://twitter.com/CatalystTeam)
[![Telegram](https://img.shields.io/badge/channel-telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-devs/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)


</div>

PyTorch framework for Deep Learning research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
- [Alchemy](https://github.com/catalyst-team/alchemy) - Experiments logging & visualization
- [Catalyst](https://github.com/catalyst-team/catalyst) - Accelerated Deep Learning Research and Development
- [Reaction](https://github.com/catalyst-team/reaction) - Convenient Deep Learning models serving

[Catalyst at AI Landscape](https://landscape.lfai.foundation/selected=catalyst).

---

# Catalyst.Classification [![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Classification_Tests/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Classification&tab=projectOverview&guest=1) [![Github contributors](https://img.shields.io/github/contributors/catalyst-team/classification.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/classification/graphs/contributors)

> *Note: this repo uses advanced Catalyst Config API and could be a bit out-of-day right now. 
> Use [Catalyst's minimal examples section](https://github.com/catalyst-team/catalyst#minimal-examples) for a starting point and up-to-day use cases, please.*

You will learn how to build image classification pipeline with transfer learning using the Catalyst framework to get reproducible results.

## Goals
1. Install requirements
2. Prepare data
3. **Run: raw data → production-ready model**
4. **Get results**
5. Customize own pipeline

## 1. Install requirements

### Using local environment:

```bash
pip install -r requirements/requirements.txt
```

### Using docker:

This creates a build `catalyst-classification` with the necessary libraries:
```bash
make docker-build
```

## 2. Get Dataset

### Try on open datasets

<details>
<summary>You can use one of the open datasets </summary>
<p>

```bash
export DATASET="artworks"

rm -rf data/
mkdir -p data

if [[ "$DATASET" == "ants_bees" ]]; then
    # https://www.kaggle.com/ajayrana/hymenoptera-data
    download-gdrive 1czneYKcE2sT8dAMHz3FL12hOU7m1ZkE7 ants_bees_cleared_190806.tar.gz
    tar -xf ants_bees_cleared_190806.tar.gz &>/dev/null
    mv ants_bees_cleared_190806 ./data/origin
elif [[ "$DATASET" == "flowers" ]]; then
    # https://www.kaggle.com/alxmamaev/flowers-recognition
    download-gdrive 1rvZGAkdLlbR_MEd4aDvXW11KnLaVRGFM flowers.tar.gz
    tar -xf flowers.tar.gz &>/dev/null
    mv flowers ./data/origin
elif [[ "$DATASET" == "artworks" ]]; then
    # https://www.kaggle.com/ikarus777/best-artworks-of-all-time
    download-gdrive 1eAk36MEMjKPKL5j9VWLvNTVKk4ube9Ml artworks.tar.gz
    tar -xf artworks.tar.gz &>/dev/null
    mv artworks ./data/origin
fi

```

</p>
</details>


### Use your own dataset


<details>
<summary>Prepare your dataset</summary>
<p>

#### Data structure
Make sure, that final folder with data has the required structure:
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

#### Data location

* The easiest way is to move your data:
    ```bash
    mv /path/to/your_dataset/* /catalyst.classification/data/origin
    ```
    In that way you can run pipeline with default settings.

* If you prefer leave data in `/path/to/your_dataset/`
    * In local environment:
        * Link directory
            ```bash
            ln -s /path/to/your_dataset $(pwd)/data/origin
            ```
         * Or just set path to your dataset `DATADIR=/path/to/your_dataset` when you start the pipeline.

    * Using docker

        You need to set:
        ```bash
           -v /path/to/your_dataset:/data \ #instead default  $(pwd)/data/origin:/data
         ```
        in the script below to start the pipeline.
</p>
</details>

## 3. Classification pipeline
### Fast&Furious: raw data → production-ready model

The pipeline will automatically guide you from raw data to the production-ready model.

We will initialize ResNet-18 model with a pre-trained network. During current pipeline model will be trained sequentially in two stages, also in the first stage we will train several heads simultaneously.

#### Run in local environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-classification-pipeline.sh \
  --workdir ./logs \
  --datadir ./data/origin \
  --max-image-size 224 \  # 224 or 448 works good
  --balance-strategy 256 \  # images in epoch per class, 1024 works good
  --config-template ./configs/templates/main.yml \
  --num-workers 4 \
  --batch-size 256 \
  --criterion CrossEntropyLoss  # one of CrossEntropyLoss, BCEWithLogits, FocalLossMultiClass
```

#### Run in docker:

```bash
docker run -it --rm --shm-size 8G --runtime=nvidia \
  -v $(pwd):/workspace/ \
  -v $(pwd)/logs:/logdir/ \
  -v $(pwd)/data/origin:/data \
  -e "CUDA_VISIBLE_DEVICES=0" \
  -e "CUDNN_BENCHMARK='True'" \
  -e "CUDNN_DETERMINISTIC='True'" \
  catalyst-classification ./bin/catalyst-classification-pipeline.sh \
    --workdir /logdir \
    --datadir /data \
    --max-image-size 224 \  # 224 or 448 works good
    --balance-strategy 256 \  # images in epoch per class, 1024 works good
    --config-template ./configs/templates/main.yml \
    --num-workers 4 \
    --batch-size 256 \
    --criterion CrossEntropyLoss  # one of CrossEntropyLoss, BCEWithLogits, FocalLossMultiClass
```
The pipeline is running and you don’t have to do anything else, it remains to wait for the best model!

#### Visualizations

You can use [W&B](https://www.wandb.com/) account for visualisation right after `pip install wandb`:

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```
<img src="/pics/wandb_metrics.png" title="w&b classification metrics"  align="left">

Tensorboard also can be used for visualisation:

```bash
tensorboard --logdir=/catalyst.classification/logs
```
<img src="/pics/tf_metrics.png" title="tf classification metrics"  align="left">

<details>
<summary>Confusion matrix</summary>
<p>
<img src="/pics/cm.png" title="tf classification metrics" width="700">
</p>
</details>

## 4. Results
All results of all experiments can be found locally in `WORKDIR`, by default `catalyst.classification/logs`. Results of experiment, for instance `catalyst.classification/logs/logdir-191010-141450-c30c8b84`, contain:

#### checkpoints
*  The directory contains all checkpoints: best, last, also of all stages.
* `best.pth` and `last.pht` can be also found in the corresponding experiment in your W&B account.

#### configs
*  The directory contains experiment\`s configs for reproducibility.

#### logs
* The directory contains all logs of experiment.
* Metrics also logs can be displayed in the corresponding experiment in your W&B account.

#### code
*  The directory contains code on which calculations were performed. This is necessary for complete reproducibility.

## 5. Customize own pipeline

For your future experiments framework provides powerful configs allow to optimize configuration of the whole pipeline of classification in a controlled and reproducible way.

<details>
<summary>Configure your experiments</summary>
<p>

* Common settings of stages of training and model parameters can be found in `catalyst.classification/configs/_common.yml`.
    * `model_params`: detailed configuration of models, including:
        * model, for instance `MultiHeadNet`
        * detailed architecture description
        * using pretrained model
    * `stages`: you can configure training or inference in several stages with different hyperparameters. In our example:
        * optimizer params
        * first learn the head(s), then train the whole network

* The `CONFIG_TEMPLATE` with other experiment\`s hyperparameters, such as data_params and is here: `catalyst.classification/configs/templates/main.yml`.  The config allows you to define:
    * `data_params`: path, batch size, num of workers and so on
    * `callbacks_params`: Callbacks are used to execute code during training, for example, to get metrics or save checkpoints. Catalyst provide wide variety of helpful callbacks also you can use custom.


You can find much more options for configuring experiments in [catalyst documentation.](https://catalyst-team.github.io/catalyst/)

</p>
</details>

## 6. Autolabel

#### Goals

The classical way to reduce the amount of unlabeled data by having a trained model would be to run unlabeled dataset through the model and automatically label images with confidence of label prediction above the threshold. Then automatically labeled data pushing in the training process so as to optimize prediction accuracy.

To run the iteration process we need to specify number of iterations `n-trials` and `threshold` of confidence to label image.

- tune ResNetEncoder
- train MultiHeadNet for image classification
- predict unlabelled dataset
- use most confident predictions as true labels
- repeat


#### Preparation

```bash
catalyst.classification/data/
    raw/
        all/
            ...
    clean/
        0/
            ...
        1/
            ...
```

#### Model training

##### Run in local environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-autolabel-pipeline.sh \
  --workdir ./logs \
  --datadir-clean ./data/clean \
  --datadir-raw ./data/raw \
  --n-trials 10 \
  --threshold 0.8 \
  --config-template ./configs/templates/autolabel.yml \
  --max-image-size 224 \
  --num-workers 4 \
  --batch-size 256
```

##### Run in docker:

```bash
docker run -it --rm --shm-size 8G --runtime=nvidia \
  -v $(pwd):/workspace/ \
  -e "CUDA_VISIBLE_DEVICES=0" \
  -e CUDNN_BENCHMARK="True" \
  -e CUDNN_DETERMINISTIC="True" \
  catalyst-classification bash ./bin/catalyst-autolabel-pipeline.sh \
    --workdir ./logs \
    --datadir-clean ./data/clean \
    --datadir-raw ./data/raw \
    --n-trials 10 \
    --threshold 0.8 \
    --config-template ./configs/templates/autolabel.yml \
    --max-image-size 224 \
    --num-workers 4 \
    --batch-size 256
```

#### Results of autolabeling
Out:
```
Predicted: 23 (100.00%)
...
Pseudo Lgabeling done. Nothing more to label.
```
Logs for trainings visualisation can be found here: `./logs/autolabel`

Labeled raw data can be found here: `/data/data_clean/dataset.csv`
