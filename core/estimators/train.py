from comet_ml import Experiment

import torch

from fastai.train import Learner, DataBunch, \
    ReduceLROnPlateauCallback, \
    EarlyStoppingCallback, \
    SaveModelCallback
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders, get_transform

from models.resnet import *
from models.omar_cnn import OmarCNN

import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

params = {'epochs': 50,
          'lr': 0.001,
          'batch_size': 128,
          'model': 'omar',
          'dataset': '100-100-0.09',
          'sampler': 'None',
          'callbacks': '[ReduceLROnPlateauCallback]',
          'resize': 32}

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

model = OmarCNN()
# model = TraversabilityResnet(1, block=BasicBlock, blocks=[2, 2, 2, 2],
#                              preactivated=False)


# model = resnet34(1, resnet=TinyResnet, n_classes=2)
# model.layers.requires_grad = False

print(model)

criterion = CrossEntropyFlat()

train_dl, val_dl, test_dl = get_dataloaders(train_root='/home/francesco/Desktop/data/train/dataset/{}'.format(params['dataset']),
                                    test_root='/home/francesco/Desktop/data/test/dataset/{}'.format(params['dataset']),
                                    val_size=0.15,
                                    transform=get_transform(params['resize']),
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

model_name_loss = '{}-{}-{}-{}-loss'.format(params['model'], params['dataset'], params['lr'],  params['resize'])
model_name_acc = '{}-{}-{}-{}-accuracy'.format(params['model'], params['dataset'], params['lr'],  params['resize'])

try:
    with experiment.train():
        learner.fit(epochs=params['epochs'], lr=params['lr'], callbacks=[
            ReduceLROnPlateauCallback(learn=learner, patience=2),
            EarlyStoppingCallback(learn=learner, patience=5),
            SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'),
            SaveModelCallback(learn=learner, name=model_name_loss),
        ]) # SaveModelCallback load the best model after training!
        # learner.fit(epochs=4, lr=params['lr'], callbacks=[ReduceLROnPlateauCallback(learn=learner, patience=1)])
        # model.layers.requires_grad = True
        # learner.fit(epochs=10, lr=params['lr'], callbacks=[ReduceLROnPlateauCallback(learn=learner, patience=1)])
except:
    pass


with experiment.test():
    loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
    print(loss, acc)
    experiment.log_metric("accuracy", acc.item())

learner = learner.load(model_name_acc)

with experiment.test():
    loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
    print(loss, acc)
    experiment.log_metric("accuracy-from-best-acc", acc.item())
