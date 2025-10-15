import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

def get_model(model_name='resnet18', num_classes=2):
    if model_name.startswith('resnet'):
        model = getattr(models, model_name)(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith('efficientnet'):
        model = EfficientNet.from_name(model_name)
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    else:
        raise ValueError("Unknown model name")
    return model
