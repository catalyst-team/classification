from typing import List, Dict
import random
import numpy as np
import cv2

import albumentations as albu
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, LongestMaxSize, PadIfNeeded,
    Normalize, HueSaturationValue, ShiftScaleRotate, RandomGamma,
    IAAPerspective, JpegCompression, ToGray, ChannelShuffle, RGBShift, CLAHE,
    RandomBrightnessContrast, RandomSunFlare, Cutout, OneOf
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
        rotate_probability: float = 1.,
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


class BlurMixin:
    """
    Calculates blur factor for augmented image
    """
    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "blur_factor",
        blur_min: int = 3,
        blur_max: int = 9,
        blur: List[str] = None
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

        self.blur_min = blur_min
        self.blur_max = blur_max
        blur = blur or ["Blur"]
        self.blur = [albu.__dict__[x]() for x in blur]
        self.num_blur = len(self.blur)
        self.num_blur_classes = blur_max - blur_min + 1 + 1
        self.blur_probability = \
            (self.num_blur_classes - 1) / self.num_blur_classes

    def __call__(self, dct):
        image = dct[self.input_key]
        blur_factor = 0

        if random.random() < self.blur_probability:
            blur_fn = np.random.choice(self.blur)
            blur_factor = int(
                np.random.randint(self.blur_min, self.blur_max) -
                self.blur_min + 1
            )
            image = blur_fn.apply(image=image, ksize=blur_factor)

        dct[self.input_key] = image
        dct[self.output_key] = blur_factor

        return dct


class FlareMixin:
    """
    Calculates blur factor for augmented image
    """
    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "flare_factor",
        sunflare_params: Dict = None
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

        self.sunflare_params = sunflare_params or {}
        self.transform = RandomSunFlare(**self.sunflare_params)

    def __call__(self, dct):
        image = dct[self.input_key]
        sunflare_factor = 0

        if random.random() < self.transform.p:
            params = self.transform.get_params()
            image = self.transform.apply(image=image, **params)
            sunflare_factor = 1

        dct[self.input_key] = image
        dct[self.output_key] = sunflare_factor

        return dct


class DictTransformCompose:
    def __init__(self, dict_transforms: List):
        self.dict_transforms = dict_transforms

    def __call__(self, dct):
        for transform in self.dict_transforms:
            dct = transform(dct)
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


def hard_transform(image_size=224, p=0.5):
    transforms = [
        Cutout(
            num_holes=4,
            max_w_size=image_size // 4,
            max_h_size=image_size // 4,
            p=p
        ),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=p
        ),
        IAAPerspective(scale=(0.02, 0.05), p=p),
        OneOf(
            [
                HueSaturationValue(p=p),
                ToGray(p=p),
                RGBShift(p=p),
                ChannelShuffle(p=p),
            ]
        ),
        RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, p=p
        ),
        RandomGamma(p=p),
        CLAHE(p=p),
        JpegCompression(quality_lower=50, p=p),
    ]
    transforms = Compose(transforms)
    return transforms
