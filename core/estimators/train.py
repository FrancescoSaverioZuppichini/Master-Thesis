from comet_ml import Experiment

import torch
import torch.nn as nn

from fastai.train import Learner, DataBunch, LearnerCallback, Recorder, MixedPrecision
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders

from models.resnet import *
from models.omar_cnn import OmarCNN

import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

params = {'epoches': 12,
          'lr': 0.0001,
          'batch_size': 128,
          'model': 'resnet18-pretrained',
          'dataset': '100-80-0.09',
          'resize': '80'}

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

# model = OmarCNN()
# model = TraversabilityResnet(1, block=BasicBlock, blocks=[2, 2, 2, 2],
#                              preactivated=False).cuda()


model = resnet18(1, resnet=TraversabilityResnet, pretrained=True)
model.layers.requires_grad = False

print(model)

criterion = CrossEntropyFlat()

train_dl, val_dl, test_dl = get_dataloaders(train_root='/home/francesco/Desktop/data/train/dataset/new-medium',
                                    test_root='/home/francesco/Desktop/data/test/dataset/{}'.format(params['dataset']),
                                    val_size=0.15,
                                    batch_size=params['batch_size'],
                                    num_workers=16,
                                    pin_memory=True)

data = DataBunch(train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl)


print("train size={}, val size={}, test size={}".format(
    len(train_dl) * params['batch_size'],
    len(val_dl) * params['batch_size'],
    len(test_dl) * params['batch_size']
))

experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                        project_name="master-thesis",
                        workspace="francescosaveriozuppichini")

experiment.log_parameters(params)

learner = Learner(data=data,
                  model=model,
                  model_dir='/home/francesco/Desktop/carino/vaevictis/data/',
                  loss_func=criterion,
                  metrics=[accuracy])
#
try:
    with experiment.train():
        # learner.fit(epochs=params['epoches'], lr=params['lr'])
        learner.fit(epochs=2, lr=params['lr'])
        model.layers.requires_grad = True
        learner.fit(epochs=10, lr=params['lr']/10)
    #
except:
    pass

learner.save(params['model'])

with experiment.test():
    loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
    print(loss, acc)
    experiment.log_metric("accuracy", acc.item())
