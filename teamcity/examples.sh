#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

bash ./teamcity/classification.sh
bash ./teamcity/autolabel.sh
