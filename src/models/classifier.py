from torch import nn

from src.models.base import Encoder


class Classifier(nn.Module):
    """Encoder + (Dropout -> Linear) head."""

    def __init__(self, encoder: Encoder, num_classes: int = 200, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.feature_dim, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.encoder.frozen:
            # A frozen encoder must not update BatchNorm running stats.
            for m in self.encoder.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self
