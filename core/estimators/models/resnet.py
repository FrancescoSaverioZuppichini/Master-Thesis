import torch
import torch.nn as nn
from torchvision.models import resnet34

model = resnet34(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=1) # binary classification

print(model)