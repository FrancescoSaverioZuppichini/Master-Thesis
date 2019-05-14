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

torch.backends.cudnn.benchmark = True


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = zoo[params['model']]
    # print(model)
    summary(model.cuda(), (1, 76,76))

    pprint.pprint(params)

    criterion = CrossEntropyFlat() if params['tr'] is not None else MSELossFlat()

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root = '/media/francesco/saetta/krock-dataset/train/',
        hm_root= '/home/francesco/Documents/Master-Thesis/core/maps/train/',
        time_window=params['time_window'],
        patch_size=params['patch_size'],
        test_root='/media/francesco/saetta/krock-dataset/test/',
        test_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/test/',
        val_root=params['validation'],
        # val_root='/media/francesco/saetta/krock-dataset/val/',
        val_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/val/',
        generate=False,
        val_size = params['val_size'],
        train_transform=get_transform(should_aug=params['data-aug']),
        val_transform=get_transform(scale=1, should_aug=False),
        test_transform=get_transform(scale=1, should_aug=False),
        num_samples=params['num_samples'],
        batch_size=params['batch_size'],
        num_workers=16,
        tr=params['tr'],
        more_than=params['more_than'],
        down_sampling=params['down_sampling'],
        only_forward=params['only_forward'],
        pin_memory=True)

    timestamp = time.time()
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

    experiment = Experiment(api_key="<HIDDEN>",
                            project_name="krock", workspace="francescosaveriozuppichini")

    experiment.log_parameters(params)
    experiment.log_metric("timestamp", timestamp)

    metrics = []

    if params['tr'] is not None: metrics = [accuracy, ROC_AUC()]

    learner = Learner(data=data,
                      model=model,
                      path=model_dir,
                      model_dir=model_dir,
                      loss_func=criterion,
                      opt_func=partial(torch.optim.Adam),
                      metrics=[*metrics, Timer()])

    model_name_roc_auc = 'roc_auc'
    model_name_acc = 'accuracy'
    model_name_loss = 'loss'

    callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=3, factor=0.2),
                 EarlyStoppingCallback(learn=learner, patience=8),
                 CSVLogger(learn=learner),
                 SaveModelCallback(learn=learner, name=model_name_loss)]

    if params['tr'] is not None:
        callbacks.append(SaveModelCallback(learn=learner, name=model_name_roc_auc, monitor='roc_auc'))
        callbacks.append(SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'))
        # callbacks.append(SaveModelCallback(learn=learner, name=model_name_loss))

    if train:
        with experiment.train():
            # learner.lr_find()
            # learner.recorder.plot()
            # plt.show() # 1e-01
            lr = 1e-3
            learner.fit_one_cycle(params['epochs'], slice(lr), pct_start=0.8, callbacks=callbacks)
            # lr =  1e-4,
            # learner.fit_one_cycle(10, slice(lr), pct_start=0.8, callbacks=callbacks)
            #
            # learner.fit(epochs=params['epochs'], lr=params['lr'],
            #             callbacks=callbacks)  # SaveModelCallback load the best model after training!
    if params['tr'] is not None:
        with experiment.test():
            learner.load(model_name_loss)
            loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc)
            experiment.log_metric("roc_auc_from_loss", roc.item())
            experiment.log_metric("test_loss_from_loss", loss)

        with experiment.test():
            learner.load(model_name_acc)
            loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc)
            experiment.log_metric("roc_auc_from_acc", roc.item())
            experiment.log_metric("test_loss_from_acc", loss)

        with experiment.test():
            learner.load(model_name_roc_auc)
            loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            print(loss, acc, roc)
            experiment.log_metric("roc_auc-from-best", roc.item())
            experiment.log_metric("test_loss_from_roc_auc", loss)

    else:
        with experiment.test():
            learner.load(model_name_loss)
            loss = learner.validate(data.test_dl)[0]
            print(loss)
            experiment.log_metric("test_acc_from_loss", loss)

    print(model_name)
    del learner.model
    del learner

if __name__ == '__main__':
    params = {'epochs': 30,
              'lr': 1e-3,
              'batch_size': 128,
              # 'model': 'omar',
              'val_size' : 10,
              'validation': None,
              'model': 'microresnet#4-gate=3x3-n=2-se=False',
              'dataset': '',
              'sampler': 'imbalance',
              'num_samples': True,
              'sampler_type': 'random',
              'data-aug': True,
              'data-aug-type': 'coarse-dropout[0.6,0.8]',
              'optim': 'adam',
              'info': '',
              'tr': 0.2,
              'problem' : 'classification',
              'more_than': 0,
              'down_sampling': 2,
              'time_window': 50 * 2,
              'only_forward': False,
              'patch_size': 0.66 }

    # params['data-aug'] = False
    #
    # for _ in range(5):
    #     train_and_evaluate(params)
    #
    # params['more_than'] = 0
    #
    # for _ in range(5):
    #     train_and_evaluate(params)

    params['data-aug'] = True
    params['data-aug-type'] = 'Dropout(p=(0.05, 0.1))-CoarseDropout((0.02, 0.1),(0.6, 0.8))-RandomSimplexNoise(2000)(1, 50)-0.8-imbalance',
    # -RandomSimplexNoise(1, 50)(4, 8)

    # for _ in range(5):
    #     train_and_evaluate(params)

    params['validation'] = '/media/francesco/saetta/krock-dataset/val/'
    # for _ in range(5):
    #     train_and_evaluate(params)
    #
    # params['model'] = 'microresnet#4-gate=3x3-n=2-se=True'
    # for _ in range(5):
    #     train_and_evaluate(params)


    params['model'] = 'microresnet#4-gate=3x3-n=1-se=True'
    params['lr'] =  1e-4
    for _ in range(5):
        train_and_evaluate(params)

