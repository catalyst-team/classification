#!/usr/bin/env bash

if [[ -z "$NUM_CLASSES" ]]
then
      NUM_CLASSES=2
fi

NUM_CLASS_ROTATION=$(($NUM_CLASSES * 8))

if [[ "$(uname)" == "Darwin" ]]; then
    sed -i ".bak" "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits.yml
    sed -i ".bak" "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_bce.yml
    sed -i ".bak" "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_focal.yml
    sed -i ".bak" "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_rotation.yml
    sed -i ".bak" "s/class_rotation_logits: \&num_class_rotations .*/class_rotation_logits: \&num_class_rotations $NUM_CLASS_ROTATION/g" ./configs/exp_splits_rotation.yml
    sed -i ".bak" "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_rotation_focal.yml
    sed -i ".bak" "s/class_rotation_logits: \&num_class_rotations .*/class_rotation_logits: \&num_class_rotations $NUM_CLASS_ROTATION/g" ./configs/exp_splits_rotation_focal.yml
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    sed -i "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits.yml
    sed -i "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_bce.yml
    sed -i "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_focal.yml
    sed -i "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_rotation.yml
    sed -i "s/class_rotation_logits: \&num_class_rotations .*/class_rotation_logits: \&num_class_rotations $NUM_CLASS_ROTATION/g" ./configs/exp_splits_rotation.yml
    sed -i "s/logits: \&num_classes .*/logits: \&num_classes $NUM_CLASSES/g" ./configs/exp_splits_rotation_focal.yml
    sed -i "s/class_rotation_logits: \&num_class_rotations .*/class_rotation_logits: \&num_class_rotations $NUM_CLASS_ROTATION/g" ./configs/exp_splits_rotation_focal.yml
fi
