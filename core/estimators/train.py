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
    SaveModelCallback, DatasetType, CSVLogger
import fastai
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat, MSELossFlat
from estimators.data import get_dataloaders, get_transform, TraversabilityDataset
from estimators.models import zoo
from estimators.callbacks import ROC_AUC, Timer
from estimators.data.transformations import RandomSimplexNoise, DropoutAgumentation
from estimators.data.dataloaders import ImbalancedDatasetSampler
from utilities.postprocessing.utils import KrockPatchExtractStrategy
torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = zoo[params['model']]()

    # print(model)

    patch_size = KrockPatchExtractStrategy.patch_shape(params['patch_size'], 0.02)
    summary(model.cuda(), (1, patch_size[0], patch_size[1]))
    # print(model)
    pprint.pprint(params)
    timestamp = time.time()

    criterion = CrossEntropyFlat() if params['tr'] is not None else MSELossFlat()

    experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                            project_name="krock-new-train", workspace="francescosaveriozuppichini")

    experiment.log_parameters(params)
    experiment.log_metric("timestamp", timestamp)

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root = '/media/francesco/saetta/krock-dataset/train/',
        hm_root= '/home/francesco/Documents/Master-Thesis/core/maps/new-train/',
        patch_size=params['patch_size'],
        test_root='/media/francesco/saetta/krock-dataset/test/',
        test_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/test/',
        val_root=params['validation'],
        # val_root='/media/francesco/saetta/krock-dataset/val/',
        val_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/val/' if params['validation'] is not None else None,
        tr=params['tr'],
        time_window=params['time_window'],
        train_transform=get_transform(aug=params['data-aug'][0]),
        train_transform_with_label=params['data-aug'][1],
        val_transform=get_transform(),
        test_transform=get_transform (),
        val_size=params['val_size'],
        batch_size=params['batch_size'],
        more_than=params['more_than'],
        less_than=params['less_than'],
        down_sampling=params['down_sampling'],
        sampler=params['sampler'],
        num_samples=params['num_samples'],
        generate=False,
        num_workers=16,
        pin_memory=True)

    model_name = '{}-{}-{}-{}-{}'.format(params['model'], params['dataset'].split('/')[0], params['lr'],
                                         params['patch_size'], timestamp)

    model_name = load_model if load_model is not None else model_name

    model_dir = path.normpath('/media/francesco/Carino/vaevictis/data/' + model_name + '/')

    os.makedirs(model_dir, exist_ok=True)

    print("train size={}, val size={}, test size={}".format(
        len(train_dl) * params['batch_size'],
        len(val_dl) * params['batch_size'],
        len(test_dl) * params['batch_size']
    ))

    data = DataBunch(train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl)

    metrics = []

    if params['tr'] is not None: metrics = [accuracy, ROC_AUC()]

    learner = Learner(data=data,
                      model=model,
                      path=model_dir,
                      model_dir=model_dir,
                      loss_func=criterion,
                      opt_func=params['optim'],
                      # opt_func=partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      # opt_func=partial(torch.optim.Adam),
                      metrics=[*metrics, Timer()])

    model_name_roc_auc = 'roc_auc'
    model_name_acc = 'accuracy'
    model_name_loss = 'loss'

    callbacks = [
                 ReduceLROnPlateauCallback(learn=learner, patience=5, factor=0.2, monitor='roc_auc'),
                 EarlyStoppingCallback(learn=learner, patience=8, monitor='roc_auc'),
                 CSVLogger(learn=learner),
                 # SaveModelCallback(learn=learner, name=model_name_loss)
    ]

    if params['tr'] is not None:
        callbacks.append(SaveModelCallback(learn=learner, name=model_name_roc_auc, monitor='roc_auc'))
        # callbacks.append(SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'))
        # callbacks.append(SaveModelCallback(learn=learner, name=model_name_loss))

    if train:
        with experiment.train():
            # learner.lr_find()
            # learner.recorder.plot()
            # plt.show() # 1e-01
            # # lr = 1e-3
            # learner.fit_one_cycle(params['epochs'], slice(params['lr']), pct_start=0.8, callbacks=callbacks)
            # # lr =  1e-4,
            # learner.fit_one_cycle(5, slice(params['lr']), pct_start=0.8, callbacks=callbacks)
            # loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            # print(loss, acc, roc)
            #
            # # learner.lr_find()
            # # learner.recorder.plot()
            # # plt.show() # 1e
            # learner.fit_one_cycle(5, slice(1e-5), callbacks=callbacks)
            # loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            # print(loss, acc, roc)
            # # #
            # learner.fit_one_cycle(5, slice(1e-6), pct_start=0.8, callbacks=callbacks)

            #
            learner.fit(epochs=params['epochs'], lr=params['lr'],
                        callbacks=callbacks)  # SaveModelCallback load the best model after training!

    if params['tr'] is not None:
        with experiment.test():
            learner.load(model_name_roc_auc)
            learner.model.eval()
            loss, acc, roc = learner.validate(data.valid_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc, 'validation')
            loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc)
            experiment.log_metric("roc_auc", roc.item())
            experiment.log_metric("acc", acc.item())

    else:
        with experiment.test():
            learner.load(model_name_loss)
            loss = learner.validate(data.test_dl)[0]
            print(loss)
            experiment.log_metric("test_acc_from_loss", loss)

    print(model_name)
    del learner.model
    del learner
    del model

if __name__ == '__main__':
    def get_params():
        return {'epochs': 30 ,
                      'lr': 1e-3,
                      'batch_size': 128,
                      'val_size' : 10,
                      'validation': None,
                      'model': 'microresnet#4-gate=3x3-n=1-se=False',
                      'dataset': '',
                      'sampler': None,
                      'num_samples': None,
                      'data-aug': (None,None),
                      # 'data-aug': None,
                      'optim': partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      # 'optim':  torch.optim.Adam,
                      'info': '',
                      'tr': 0.2,
                      'problem' : 'classification',
                      'more_than': None,
                      'less_than': None,
                      'down_sampling': 2,
                      'time_window': 50 * 2,
                      'patch_size': 0.71
                      }

    params = get_params()

    params = get_params()
    params['epochs'] = 30
    params['validation'] = '/media/francesco/saetta/krock-dataset/val/'
    # opt_func=partial(torch.optim.Adam),
    params['model'] = 'microresnet#4-gate=3x3-n=1-se=True'
    # params['sampler'] = ImbalancedDatasetSampler
    # params['num_samples'] = 1000
    # params['data-aug'] = (DropoutAgumentation(), RandomSimplexNoise(n=1000, p=0.8))
    for _ in range(2):
        train_and_evaluate(params)


