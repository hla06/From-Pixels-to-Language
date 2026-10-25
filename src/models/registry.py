"""Encoder registry: a Factory keyed by name.

Encoders self-register with @register(name); callers only use build_encoder,
so adding an encoder never touches the training loop.
"""
from typing import Callable

from src.models.base import Encoder

_REGISTRY: dict[str, Callable[[bool], Encoder]] = {}


def register(name: str):
    def deco(builder):
        _REGISTRY[name] = builder
        return builder

    return deco


def build_encoder(name: str, pretrained: bool = True, freeze: bool = False) -> Encoder:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown encoder '{name}'. Known: {sorted(_REGISTRY)}")
    enc = _REGISTRY[name](pretrained)
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
        enc.frozen = True
    return enc


# Imported last so the modules can use `register` without a circular-import error.
from src.models import scratch_cnn, torchvision_encoders  # noqa: E402,F401
