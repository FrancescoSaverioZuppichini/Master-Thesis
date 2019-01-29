from comet_ml import Experiment

import torch
import torch.nn as nn

from fastai.train import Learner, DataBunch, LearnerCallback
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders

from models.resnet import *
from models.omar_cnn import OmarCNN

torch.backends.cudnn.benchmark = True

torch.backends.cudnn.deterministic = True

torch.manual_seed(0)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

# model = OmarCNN()
model = TraversabilityResnet(1, block=BasicBlock, blocks=[1, 2, 3, 2], preactivated=True)

criterion = CrossEntropyFlat()

train_dl, test_dl = get_dataloaders()

data = DataBunch(train_dl=train_dl, valid_dl=test_dl)

params = {'epoches': 20,
          'lr': 0.0005,
          'batch_size': 128,
          'preactivated': True,
          'model': 'resnet-tiny',
          'dataset': 'tiny',
          'resize': '80'}

print(model)
experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                        project_name="master-thesis", workspace="francescosaveriozuppichini")

experiment.log_parameters(params)

learner = Learner(data=data,
                  model=model,
                  loss_func=criterion,
                  metrics=[accuracy])

with experiment.train():
    learner.fit(epochs=params['epoches'], lr=params['lr'])
