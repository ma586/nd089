import torch
import torch.nn as nn
from torchvision import models

class FlowerModel(nn.Module):

    def __init__(self, hidden_units:int , arch: str , device: str , class_to_idx = None):
        super().__init__()

        self.backbone = models.get_model(arch, weights=True)

        for p in self.parameters():
            p.requires_grad = False

        nr_of_features = self.backbone.fc.in_features #nr of categories
        self.hidden_units = hidden_units

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(nr_of_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_units, nr_of_features),
            torch.nn.LogSoftmax(dim=1)
        )
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx

        self.backbone.fc = self.classifier

        self.to(device)

    def forward(self, x):
        return self.backbone(x)