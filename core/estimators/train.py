from comet_ml import Experiment

import torch
import torch.nn as nn

from fastai.train import Learner, DataBunch, LearnerCallback
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders

from models.resnet import *
from models.omar_cnn import OmarCNN

from callbacks import CometCallback

torch.backends.cudnn.benchmark = True

# model = OmarCNN()
model = TraversabilityResnet(1, block=BasicBlock, blocks=[3, 4, 6, 3], preactivated=True)

criterion = CrossEntropyFlat()

train_dl, test_dl = get_dataloaders()

data = DataBunch(train_dl=train_dl, valid_dl=test_dl)

params = { 'epoches': 20,
           'lr': 0.001,
           'batch_size': 64,
           'preactivated' : True,
           'model': 'Resnet34'}

print(model)
experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                        project_name="master-thesis", workspace="francescosaveriozuppichini")

experiment.log_parameters(params)

learner = Learner(data=data,
                  model=model,
                  loss_func=criterion,
                  metrics=[accuracy]).to_fp16()

with experiment.train():

    learner.fit(epochs=100, lr=0.0001)