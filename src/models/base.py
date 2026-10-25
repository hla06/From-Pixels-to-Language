from torch import nn


class Encoder(nn.Module):
    """Contract every backbone follows, so tasks can swap encoders freely.

    forward(x) returns [B, feature_dim]; gradcam_layer() returns the module
    whose activations Grad-CAM should use. `frozen` is set by build_encoder.
    """

    feature_dim: int
    frozen: bool = False

    def gradcam_layer(self) -> nn.Module:
        raise NotImplementedError
