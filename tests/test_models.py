import pytest
import torch
from torch import nn

from src.models.classifier import Classifier
from src.models.registry import build_encoder

DIMS = {"scratch_cnn": 512, "resnet18": 512, "resnet50": 2048, "efficientnet_b0": 1280}


@pytest.mark.parametrize("name", list(DIMS))
def test_output_shape_and_feature_dim(name):
    enc = build_encoder(name, pretrained=False).eval()
    assert enc.feature_dim == DIMS[name]
    with torch.no_grad():
        assert enc(torch.randn(2, 3, 224, 224)).shape == (2, DIMS[name])


@pytest.mark.parametrize("name", list(DIMS))
def test_gradcam_layer_is_submodule(name):
    enc = build_encoder(name, pretrained=False)
    layer = enc.gradcam_layer()
    assert any(m is layer for m in enc.modules())


def test_unknown_name_lists_known():
    with pytest.raises(KeyError, match="resnet18"):
        build_encoder("nope", pretrained=False)


def test_frozen_encoder_no_grad_and_bn_eval():
    enc = build_encoder("resnet18", pretrained=False, freeze=True)
    assert not any(p.requires_grad for p in enc.parameters())
    model = Classifier(enc, num_classes=200)
    model.train()
    bns = [m for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    assert model.head[0].training


def test_frozen_efficientnet_encoder_fully_eval():
    enc = build_encoder("efficientnet_b0", pretrained=False, freeze=True)
    model = Classifier(enc, num_classes=200)
    model.train()
    assert not any(m.training for m in model.encoder.modules())  # BN, Dropout, StochasticDepth
    assert model.head[0].training


def test_unfrozen_bn_trains():
    model = Classifier(build_encoder("resnet18", pretrained=False))
    model.train()
    assert any(m.training for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d))


def test_head_shape():
    model = Classifier(build_encoder("scratch_cnn", pretrained=False), num_classes=200)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 200)
