import torch
import torch.nn as nn
from torchvision import models

class MultiModalTBModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)

        self.clinical_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, image, clinical):
        img_feat = self.cnn(image)
        clin_feat = self.clinical_fc(clinical)
        combined = torch.cat((img_feat, clin_feat), dim=1)
        return self.classifier(combined)