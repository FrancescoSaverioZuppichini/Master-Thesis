from comet_ml import Experiment

import os
import torch
import pprint
import numpy as np
import time

from functools import partial
from torch.nn.functional import softmax
from torchsummary import summary
from os import path
from fastai.train import Learner, DataBunch, \
    ReduceLROnPlateauCallback, \
    EarlyStoppingCallback, \
    SaveModelCallback, DatasetType

from fastai.callback import Callback
from fastai.metrics import accuracy, dice
from fastai.layers import CrossEntropyFlat, MSELossFlat
from datasets.TraversabilityDataset import TraversabilityDataset
from sklearn.metrics import roc_auc_score
from datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from models import zoo
from callbacks import ROC_AUC, Timer

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(0)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = zoo[params['model']]
    # print(model)
    if params['resize'] is not None: summary(model.cuda(), (1, *params['resize']))
    else: summary(model.cuda(), (1, 88, 88))

    pprint.pprint(params)

    criterion = CrossEntropyFlat() if params['tr'] is not None else MSELossFlat()

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/media/francesco/saetta/{}/train/'.format(params['dataset']),
        val_root='/media/francesco/saetta/{}/val/'.format(params['dataset']),
        test_root='/media/francesco/saetta/{}/test/'.format(params['dataset']),
        train_transform=get_transform(resize=params['resize'], should_aug=params['data-aug']),
        val_transform=get_transform(resize=params['resize'], scale=1, should_aug=False),
        test_transform=get_transform(resize=params['resize'], scale=10, should_aug=False),
        num_samples=params['num_samples'],
        batch_size=params['batch_size'],
        num_workers=16,
        tr=params['tr'],
        more_than=params['more_than'],
        downsample_factor=params['downsample_factor'],
        only_forward=params['only_forward'],
        pin_memory=True)

    timestamp = time.time()
    model_name = '{}-{}-{}-{}-{}'.format(params['model'], params['dataset'].split('/')[0], params['lr'],
                                         params['resize'], timestamp)

    model_name = load_model if load_model is not None else model_name

    model_dir = path.normpath('/home/francesco/Desktop/carino/vaevictis/data/' + model_name + '/')

    os.makedirs(model_dir, exist_ok=True)

    print("train size={}, val size={}, test size={}".format(
        len(train_dl) * params['batch_size'],
        len(val_dl) * params['batch_size'],
        len(test_dl) * params['batch_size']
    ))

    data = DataBunch(train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl)

    experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                            project_name="krock-new", workspace="francescosaveriozuppichini")

    experiment.log_parameters(params)
    experiment.log_metric("timestamp", timestamp)

    learner = Learner(data=data,
                      model=model,
                      path='/home/francesco/Desktop/carino/vaevictis/data/',
                      model_dir=model_dir,
                      loss_func=criterion,
                      opt_func=partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      metrics=[accuracy, ROC_AUC(), Timer()])

    model_name_acc = 'roc_auc'
    model_name_loss = 'loss'

    callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=4),
                 EarlyStoppingCallback(learn=learner, patience=6),
                 SaveModelCallback(learn=learner, name=model_name_acc, monitor='roc_auc'),
                 SaveModelCallback(learn=learner, name=model_name_loss)]

    if train:
        # try:
        with experiment.train():
            learner.fit(epochs=params['epochs'], lr=params['lr'],
                        callbacks=callbacks)  # SaveModelCallback load the best model after training!

        # except Exception as e:
        #     print(e)
        #     pass

        with experiment.test():
            learner.load(model_name_loss)
            loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc)
            experiment.log_metric("roc_auc", roc.item())
            experiment.log_metric("test_loss", loss)

        with experiment.test():
            learner.load(model_name_acc)
            loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc)
            experiment.log_metric("roc_auc-from-best", roc.item())

    del learner.model
    del learner

if __name__ == '__main__':
    params = {'epochs': 20,
              'lr': 0.001,
              'batch_size': 128,
              # 'model': 'omar',
              'model': 'microresnet#4-gate=3x3-n=2-se=False',
              'dataset': 'no-shift-88-750',
              'sampler': '',
              'num_samples': None,
              'samper_type': 'random',
              'callbacks': '[ReduceLROnPlateauCallback]',
              'data-aug': True,
              'optim': 'sdg',
              'info': 'aggreassive data aug-val new',
              'tr': 0.45,
              'more_than': -0.5,
              'downsample_factor': None,
              'time_window': 750,
              'only_forward': False,
              'resize': None  }

    train_and_evaluate(params)
    train_and_evaluate(params)
    train_and_evaluate(params)
    train_and_evaluate(params)
    train_and_evaluate(params)
    train_and_evaluate(params)







