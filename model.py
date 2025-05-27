# model.py
import torch
import torch.nn as nn
import timm

import config

# model = EfficientNetB4
def build_model(num_classes=config.NUM_CLASSES):
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=config.NUM_CLASSES)
    return model.to(config.DEVICE, memory_format=torch.channels_last)