from comet_ml import Experiment

import os
import torch
import pprint
import numpy as np

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

import matplotlib.pyplot as plt
from functools import partial
import time
import seaborn as sns

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(0)


class ROC_AUC(Callback):
    """
    Compute roc auc by storing all the predictions/targets and run it once
    when the epoch is over
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = np.array([])
        self.targs = np.array([])

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            preds = softmax(last_output, dim=1)
            preds = torch.argmax(preds, dim=1).long().cpu().numpy()
            targs = last_target.cpu().numpy()

            self.preds = np.append(self.preds, preds)
            self.targs = np.append(self.targs, targs)

    def on_epoch_end(self, last_metrics, last_target, last_output, **kwargs):
        roc = roc_auc_score(self.preds, self.targs)
        self.metric = roc
        self.preds = np.array([])
        self.targs = np.array([])
        return {'last_metrics': last_metrics + [torch.tensor(self.metric)]}

class Timer(Callback):
    def on_epoch_begin(self, **kwargs):
        self.start = time.time()

    def on_epoch_end(self, train, last_metrics, **kwargs):
        self.metric = time.time() - self.start

        return {'last_metrics': last_metrics + [self.metric]}


def custom_accuracy(y_pred, y_true, thresh: float = 0.01):
    # print(y_pred[0:10], y_true[0:10])
    distance = (y_pred - y_true).abs()
    acc = (distance < thresh).float().mean()
    return acc


if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = zoo[params['model']]

    summary(model.cuda(), (1, 92, 92))
    pprint.pprint(params)

    criterion = CrossEntropyFlat()

    # criterion = MSELossFlat()

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/home/francesco/Desktop/data/{}/train/df/'.format(params['dataset']),
        val_root='/home/francesco/Desktop/data/{}/val/df/'.format(params['dataset']),
        test_root='/home/francesco/Desktop/data/{}/test/df/'.format(params['dataset']),

        # val_size=0.15,
        train_transform=get_transform(params['resize'], should_aug=params['data-aug']),
        val_transform=get_transform(params['resize'], scale=1, should_aug=False),
        test_transform=get_transform(params['resize'], scale=10, should_aug=False),
        num_samples=params['num_samples'],
        batch_size=params['batch_size'],
        num_workers=16,
        tr=params['tr'],
        more_than=params['more_than'],
        downsample_factor=params['downsample_factor'],
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
        try:
            with experiment.train():
                learner.fit(epochs=params['epochs'], lr=params['lr'],
                            callbacks=callbacks)  # SaveModelCallback load the best model after training!

            learner.load(model_name_loss)

            with experiment.test():
                loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
                print(loss, acc, roc)
                experiment.log_metric("roc_auc", roc.item())
                experiment.log_metric("test_loss", loss)

            learner.load(model_name_acc)
        except Exception as e:
            print(e)
            pass

    if load_model:
        print('Loading model...')
        learner.load(model_name_acc)

    with experiment.test():
        loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
        print(loss, acc, roc)
        experiment.log_metric("roc_auc-from-best", roc.item())


params = {'epochs': 10,
          'lr': 0.001,
          'batch_size': 128,
          # 'model': 'omar',
          'model': 'microresnet#4-gate=3x3-n=2-se=True',
          'dataset': '750',
          'sampler': '',
          'num_samples': None,
          'samper_type': 'random',
          'callbacks': '[ReduceLROnPlateauCallback]',
          'data-aug': True,
          'optim': 'sdg',
          'info': 'scale before center',
          'tr': 0.45,
          'more_than': -0.5,
          'downsample_factor': None,
          'time_window': 750,
          'resize': None}


train_and_evaluate(params)
train_and_evaluate(params)
train_and_evaluate(params)

