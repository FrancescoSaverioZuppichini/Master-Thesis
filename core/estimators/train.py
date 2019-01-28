import torch
import torch.nn as nn
from fastai.train import Learner, DataBunch, LearnerCallback
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat
from datasets.TraversabilityDataset import get_dataloaders
from models.resnet import TraversabilityResnet
from models.omar_cnn import OmarCNN

# model = OmarCNN()
model = TraversabilityResnet(1, blocks=[2,2,2,2])

criterion = CrossEntropyFlat()

train_dl, test_dl = get_dataloaders()

data = DataBunch(train_dl=train_dl, valid_dl=test_dl)

print(model)

learner = Learner(data=data,
                  model=model,
                  loss_func=criterion,
                  metrics=[accuracy])

learner.fit(epochs=100, lr=0.001)