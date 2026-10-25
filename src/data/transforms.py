"""Image transforms: ImageNet normalisation everywhere so all encoders are compared fairly."""
from typing import Callable

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, image_size: int = 224, aug: dict | None = None) -> Callable:
    aug = aug or {}
    to_rgb = T.Lambda(lambda im: im.convert("RGB"))  # grayscale/RGBA files -> 3 channels
    tail = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        # resize to ~1.14x then centre crop (256 -> 224 at the default size)
        return T.Compose([to_rgb, T.Resize(round(image_size * 256 / 224)),
                          T.CenterCrop(image_size), *tail])
    jitter = aug.get("jitter", 0.2)
    return T.Compose([
        to_rgb,
        T.RandomResizedCrop(image_size, scale=(aug.get("scale_min", 0.5), 1.0)),
        T.RandomHorizontalFlip(aug.get("hflip", 0.5)),  # no vertical flip: birds are upright
        T.RandomRotation(aug.get("rotation", 15)),
        # mild jitter and tiny hue shift: colour is a species cue
        T.ColorJitter(jitter, jitter, jitter, aug.get("hue", 0.02)),
        *tail,
    ])
