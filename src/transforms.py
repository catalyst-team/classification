import random
import numpy as np
import cv2

from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, LongestMaxSize, PadIfNeeded,
    Normalize, HueSaturationValue, ShiftScaleRotate, RandomGamma,
    IAAPerspective, JpegCompression, ToGray, ChannelShuffle, RGBShift, CLAHE,
    RandomBrightnessContrast
)
from albumentations.torch import ToTensor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class RotateMixin:
    """
    Calculates rotation factor for augmented image
    """

    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "rotation_factor",
        targets_key: str = None,
        rotate_probability: float = 0.5,
        hflip_probability: float = 0.5,
        one_hot_classes: int = None
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key
        self.targets_key = targets_key
        self.rotate_probability = rotate_probability
        self.hflip_probability = hflip_probability
        self.rotate = RandomRotate90()
        self.hflip = HorizontalFlip()
        self.one_hot_classes = one_hot_classes * 8 \
            if one_hot_classes is not None \
            else None

    def __call__(self, dct):
        image = dct[self.input_key]
        rotation_factor = 0

        if random.random() < self.rotate_probability:
            rotation_factor = self.rotate.get_params()["factor"]
            image = self.rotate.apply(img=image, factor=rotation_factor)

        if random.random() < self.hflip_probability:
            rotation_factor += 4
            image = self.hflip.apply(img=image)

        dct[self.input_key] = image
        dct[self.output_key] = rotation_factor

        if self.targets_key is not None:
            class_rotation_factor = dct[self.targets_key] * 8 + rotation_factor
            dct[f"class_rotation_{self.targets_key}"] = class_rotation_factor

            if self.one_hot_classes is not None:
                one_hot = np.zeros(self.one_hot_classes, dtype=np.float32)
                one_hot[class_rotation_factor] = 1.0
                dct[f"class_rotation_{self.targets_key}_one_hot"] = one_hot

        return dct


class MixinAdapter:
    def __init__(self, mixin, pre_transforms, post_transforms):
        self.mixin = mixin
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

    def __call__(self, dct):
        dct = self.pre_transforms(dct)
        dct = self.mixin(dct)
        dct = self.post_transforms(dct)
        return dct


def pre_transforms(image_size=224):
    return Compose(
        [
            LongestMaxSize(max_size=image_size),
            PadIfNeeded(
                image_size, image_size, border_mode=cv2.BORDER_CONSTANT
            ),
        ]
    )


def post_transforms():
    return Compose([Normalize(), ToTensor()])


def hard_transform():
    transforms = [
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        IAAPerspective(scale=(0.02, 0.05), p=0.3),
        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        RandomGamma(gamma_limit=(85, 115), p=0.3),
        HueSaturationValue(p=0.3),
        ChannelShuffle(p=0.5),
        ToGray(p=0.2),
        CLAHE(p=0.3),
        RGBShift(p=0.3),
        JpegCompression(quality_lower=50),
    ]
    transforms = Compose(transforms)
    return transforms
