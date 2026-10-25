from torch import nn
from torchvision import models as tvm

from src.models.base import Encoder
from src.models.registry import register


class _ResNetEncoder(Encoder):
    def __init__(self, net: nn.Module, feature_dim: int):
        super().__init__()
        net.fc = nn.Identity()  # drop the ImageNet classifier, keep global pooling
        self.net = net
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x)

    def gradcam_layer(self):
        return self.net.layer4


class _EfficientNetEncoder(Encoder):
    def __init__(self, net: nn.Module, feature_dim: int):
        super().__init__()
        net.classifier = nn.Identity()
        self.net = net
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x)

    def gradcam_layer(self):
        return self.net.features[-1]


@register("resnet18")
def _resnet18(pretrained: bool) -> Encoder:
    w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return _ResNetEncoder(tvm.resnet18(weights=w), 512)


@register("resnet50")
def _resnet50(pretrained: bool) -> Encoder:
    w = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return _ResNetEncoder(tvm.resnet50(weights=w), 2048)


@register("efficientnet_b0")
def _efficientnet_b0(pretrained: bool) -> Encoder:
    w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    return _EfficientNetEncoder(tvm.efficientnet_b0(weights=w), 1280)
