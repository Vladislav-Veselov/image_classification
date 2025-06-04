from torchvision.models import resnet18
from torch import nn

def ResNet18(num_classes: int = 10):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
