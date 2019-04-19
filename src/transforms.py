import cv2

from albumentations import (
    RandomRotate90, Normalize, Compose, ShiftScaleRotate, JpegCompression,
    LongestMaxSize, PadIfNeeded
)
from albumentations.torch import ToTensor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def post_transform():
    return Compose([Normalize(), ToTensor()])


def train_transform(image_size=224):
    transforms = [
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        RandomRotate90(),
        JpegCompression(quality_lower=50),
        post_transform()
    ]
    transforms = Compose(transforms)
    return transforms


def valid_transform(image_size=224):
    transforms = [
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
        post_transform()
    ]
    transforms = Compose(transforms)
    return transforms
