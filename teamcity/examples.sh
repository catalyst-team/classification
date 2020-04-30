#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

pip install -r requirements/requirements.txt

# @TODO: fix server issue
pip install torch==1.4.0 torchvision==0.5.0

bash ./bin/tests/_check_classification_pipeline.sh

bash ./bin/tests/_check_autolabel_pipeline.sh
