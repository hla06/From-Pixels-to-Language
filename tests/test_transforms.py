import torch
from PIL import Image

from src.data.transforms import build_transforms


def _img(mode):
    return Image.effect_noise((300, 260), 50).convert(mode)


def test_gray_and_rgba_to_three_channels():
    for train in (True, False):
        t = build_transforms(train=train)
        for mode in ("L", "RGBA"):
            assert t(_img(mode)).shape == (3, 224, 224)


def test_eval_transform_deterministic():
    t = build_transforms(train=False)
    img = _img("RGB")
    assert torch.equal(t(img), t(img))


def test_train_transform_stochastic():
    t = build_transforms(train=True)
    img = _img("RGB")
    assert not torch.equal(t(img), t(img))
