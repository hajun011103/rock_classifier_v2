import torch
import torch.nn as nn
from torchvision import models
import config

def build_model(num_classes=config.NUM_CLASSES):
    model = models.convnext_tiny(weights="DEFAULT")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model.to(config.DEVICE, memory_format=torch.channels_last)
