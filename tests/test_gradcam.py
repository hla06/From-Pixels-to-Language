import numpy as np
import torch

from src.models.classifier import Classifier
from src.models.registry import build_encoder
from src.visualization.gradcam import GradCAM, overlay


def _model():
    torch.manual_seed(0)
    enc = build_encoder("scratch_cnn", pretrained=False)
    return Classifier(enc, num_classes=5).eval()


def test_cam_shape_and_range():
    m = _model()
    cam = GradCAM(m, m.encoder.gradcam_layer())
    out = cam(torch.randn(1, 3, 64, 64))
    cam.close()
    assert out.shape == (64, 64)
    assert out.min() >= 0 and out.max() <= 1


def test_cam_differs_by_class():
    m = _model()
    cam = GradCAM(m, m.encoder.gradcam_layer())
    x = torch.randn(1, 3, 64, 64)
    a, b = cam(x, class_idx=0), cam(x, class_idx=1)
    cam.close()
    assert not np.allclose(a, b)


def test_close_removes_hooks():
    m = _model()
    layer = m.encoder.gradcam_layer()
    cam = GradCAM(m, layer)
    cam(torch.randn(1, 3, 64, 64))
    cam.close()
    assert len(layer._forward_hooks) == 0
    assert len(layer._backward_hooks) == 0
    assert len(layer._forward_pre_hooks) == 0
    assert len(layer._backward_pre_hooks) == 0


def test_constant_cam_is_zeros_not_nan():
    m = _model()
    for p in m.parameters():
        torch.nn.init.zeros_(p)  # all activations/gradients zero -> constant cam
    cam = GradCAM(m, m.encoder.gradcam_layer())
    out = cam(torch.randn(1, 3, 64, 64))
    cam.close()
    assert not np.isnan(out).any() and out.max() == 0


def test_overlay_returns_rgb_image_of_same_size():
    from PIL import Image
    img = Image.new("RGB", (40, 30), "white")
    out = overlay(img, np.random.rand(30, 40))
    assert out.size == (40, 30) and out.mode == "RGB"
