from comet_ml import Experiment

import torch

# from torchvision.models import *

from torchsummary import summary

from fastai.train import Learner, DataBunch, \
    ReduceLROnPlateauCallback, \
    EarlyStoppingCallback, \
    SaveModelCallback
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders, get_transform, get_train_transform

from models.resnet import *
from models.omar_cnn import OmarCNN

from models.custom_resnet import *
import matplotlib.pyplot as plt

from functools import partial

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

params = {'epochs': 0,
          'lr': 0.001,
          'batch_size': 128,
          'model': 'omar',
          'dataset': '100-100-0.1-12-no_tail-spawn',
          'val_dataset': '100-100-0.09-12-no_tail',
          'test_dataset': '100-100-0.1-12-querry-no_tail-spawn',
          'sampler': None,
          'samper_type': 'sample',
          'callbacks': '[ReduceLROnPlateauCallback]',
          'data-aug': 'None',
          'optim': 'adam',
          'info': 'remove',
          'resize': 64 }

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)


model = OmarCNN()
# model = MicroResnet.micro(1,
#                           n_classes=2,
#                           block=[BasicBlock, BasicBlock, BasicBlock, BasicBlock],
#                           preactivated=True)
# print(model)


summary(model.cuda(), (1, params['resize'], params['resize']))

criterion = CrossEntropyFlat()

train_dl, val_dl, test_dl = get_dataloaders(train_root='/home/francesco/Desktop/data/train/dataset/{}'.format(params['dataset']),
                                    test_root='/home/francesco/Desktop/data/test/dataset/{}'.format(params['test_dataset']),
                                    # val_root='/home/francesco/Desktop/data/val/dataset/{}'.format(params['val_dataset']),
                                    val_size=0.15,
                                    transform=get_transform(params['resize'], scale=10),
                                    train_transform=get_train_transform(params['resize']),
                                    num_samples=params['sampler'],
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
                  path='/home/francesco/Desktop/carino/vaevictis/data/',
                  model_dir='/home/francesco/Desktop/carino/vaevictis/data/',
                  loss_func=criterion,
                  # opt_func= partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                  metrics=[accuracy])

model_name_acc = '{}-{}-{}-{}-accuracy'.format(params['model'], params['dataset'], params['lr'],  params['resize'])
model_name_loss = '{}-{}-{}-{}-loss'.format(params['model'], params['dataset'], params['lr'],  params['resize'])

callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=3),
            EarlyStoppingCallback(learn=learner, patience=5),
            SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'),
            SaveModelCallback(learn=learner, name=model_name_loss)]
try:
    with experiment.train():
        learner.fit(epochs=params['epochs'], lr=params['lr'], callbacks=callbacks) # SaveModelCallback load the best model after training!
except Exception as e:
    print(e)
    pass

torch.save(learner.model, '/home/francesco/Desktop/carino/vaevictis/data/{}.pck'.format(params['model']))


learner.load(model_name_loss)


with experiment.test():
    loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
    print(loss, acc)
    experiment.log_metric("accuracy", acc.item())
    experiment.log_metric("test_loss", loss)

learner = learner.load(model_name_acc)

torch.save(learner.model, '/home/francesco/Desktop/carino/vaevictis/data/{}.pck'.format(params['model']))

with experiment.test():
    loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
    print(loss, acc)
    experiment.log_metric("accuracy-from-best-acc", acc.item())

