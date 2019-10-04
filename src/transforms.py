import cv2

from albumentations import (
    Compose, LongestMaxSize, PadIfNeeded,
    Normalize, HueSaturationValue, ShiftScaleRotate, RandomGamma,
    IAAPerspective, JpegCompression, ToGray, ChannelShuffle, RGBShift, CLAHE,
    RandomBrightnessContrast, Cutout, OneOf
)
from albumentations.torch import ToTensor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


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
