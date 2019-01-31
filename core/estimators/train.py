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

params = {'epoches': 20,
          'lr': 0.0001,
          'batch_size': 128,
          'model': 'resnet18-pretrained=False',
          'dataset': 'medium-h-center',
          'resize': '80'}

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

# model = OmarCNN()
model = TraversabilityResnet(1, block=BasicBlock, blocks=[2, 2, 2, 2],
                             preactivated=False).cuda()


# model = resnet18(1, resnet=TraversabilityResnet, pretrained=True)
#
# model.layers.requires_grad = False

criterion = CrossEntropyFlat()

train_dl, test_dl = get_dataloaders(train_root='/home/francesco/Desktop/data/dataset/train/images-medium',
                                    test_root='/home/francesco/Desktop/data/dataset/test/images/',
                                    batch_size=params['batch_size'],
                                    num_workers=16,
                                    pin_memory=True)

data = DataBunch(train_dl=train_dl, valid_dl=test_dl)


print("train size={}, val size ={}".format(len(train_dl) * params['batch_size'], len(test_dl) * params['batch_size']))


print(model)
experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                        project_name="master-thesis",
                        workspace="francescosaveriozuppichini")

experiment.log_parameters(params)

learner = Learner(data=data,
                  model=model,
                  loss_func=criterion,
                  metrics=[accuracy])

with experiment.train():
    learner.fit(epochs=params['epoches'], lr=params['lr'])
