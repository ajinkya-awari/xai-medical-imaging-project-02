
import torch
import torch.nn as nn
from torchvision import models
from src.config import CFG


class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=CFG.NUM_CLASSES, pretrained=True):
        super().__init__()
        weights       = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base          = models.densenet121(weights=weights)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(CFG.DROPOUT),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        features = torch.relu(features)
        return self.classifier(features)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True
