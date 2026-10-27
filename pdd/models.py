"""ImageNet-pretrained backbones with a new classification head."""
import torch.nn as nn
from torchvision import models

ARCHS = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


def build(arch, num_classes, pretrained=True):
    fn, weights = ARCHS[arch]
    m = fn(weights=weights if pretrained else None)
    if arch == "resnet18":
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        last = m.classifier[-1]
        m.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())
