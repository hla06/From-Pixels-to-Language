from torch import nn

from src.models.base import Encoder
from src.models.registry import register


class ScratchCNN(Encoder):
    """Baseline: 5 x (Conv3x3-BN-ReLU-MaxPool), trained from random init."""

    def __init__(self, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        layers, in_ch = [], 3
        for w in widths:
            layers += [
                nn.Conv2d(in_ch, w, 3, padding=1, bias=False),  # BN follows, so no bias
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feature_dim = widths[-1]
        self._last_conv = self.features[-4]

    def forward(self, x):
        return self.pool(self.features(x))

    def gradcam_layer(self):
        return self._last_conv


@register("scratch_cnn")
def _build(pretrained: bool) -> Encoder:
    return ScratchCNN()  # nothing to pretrain
