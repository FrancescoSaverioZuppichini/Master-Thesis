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
import fastai
from fastai.metrics import accuracy, dice
from fastai.layers import CrossEntropyFlat, MSELossFlat
from estimators.datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from estimators.models import zoo
from estimators.callbacks import ROC_AUC, Timer

# torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# np.random.seed(0)
# if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

def custom_accuracy(y_pred, y_true, thresh: float = 0.01):
    distance = (y_pred - y_true).abs()
    acc = (distance < thresh).float().mean()
    return acc


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = zoo[params['model']]
    # print(model)
    summary(model.cuda(), (1, params['patch_size'], params['patch_size']))

    pprint.pprint(params)

    criterion = CrossEntropyFlat() if params['tr'] is not None else MSELossFlat()

    train_dl, val_dl, test_dl = get_dataloaders(
        meta_path = '/media/francesco/saetta/krock-dataset/train/bags/meta.csv',
        train_root = '/media/francesco/saetta/krock-dataset/train/csvs_parsed/',
        hm_root= '/home/francesco/Documents/Master-Thesis/core/maps/train/',
        time_window=params['time_window'],
        patch_size=params['patch_size'],
        test_meta='/media/francesco/saetta/krock-dataset/test/bags/meta.csv',
        test_root='/media/francesco/saetta/krock-dataset/test/csvs_parsed/',
        test_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/test/',
        # val_root='/media/francesco/saetta/{}/val/'.format(params['dataset']),
        val_size = params['val_size'],
        train_transform=get_transform(should_aug=params['data-aug']),
        val_transform=get_transform(scale=1, should_aug=False),
        test_transform=get_transform(scale=10, should_aug=False),
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

    model_dir = path.normpath('/home/francesco/Desktop/carino/vaevictis/data/' + model_name + '/')

    os.makedirs(model_dir, exist_ok=True)

    print("train size={}, val size={}, test size={}".format(
        len(train_dl) * params['batch_size'],
        len(val_dl) * params['batch_size'],
        len(test_dl) * params['batch_size']
    ))

    data = DataBunch(train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl)

    experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                            project_name="krock-new-new", workspace="francescosaveriozuppichini")

    experiment.log_parameters(params)
    experiment.log_metric("timestamp", timestamp)

    accuracy = fastai.metrics.accuracy if params['tr'] is not None else custom_accuracy

    learner = Learner(data=data,
                      model=model,
                      path='/home/francesco/Desktop/carino/vaevictis/data/',
                      model_dir=model_dir,
                      loss_func=criterion,
                      opt_func=partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      metrics=[accuracy, ROC_AUC(), Timer()])

    model_name_roc_auc = 'roc_auc'
    model_name_acc = 'accuracy'
    model_name_loss = 'loss'

    callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=4),
                 EarlyStoppingCallback(learn=learner, patience=6),
                 SaveModelCallback(learn=learner, name=model_name_roc_auc, monitor='roc_auc'),
                 SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'),
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



    del learner.model
    del learner

if __name__ == '__main__':
    params = {'epochs': 10,
              'lr': 0.001,
              'batch_size': 128,
              # 'model': 'omar',
              'val_size' : 10,
              'model': 'microresnet#4-gate=3x3-n=1-se=True',
              'dataset': 'no-shift',
              'sampler': '',
              'num_samples': None,
              'sampler_type': 'random',
              'data-aug': True,
              'optim': 'sgd',
              'info': '',
              'tr': 0.45,
              'more_than': 0,
              'down_sampling': 2,
              'time_window': 50 * 3,
              'only_forward': False,
              'patch_size': 88  }


    for _ in range(4):
        train_and_evaluate(params)

    # params['resize'] = None
    # params['num_samples'] = 50000
    #
    # for _ in range(10):
    #     train_and_evaluate(params)
    #
    # params['resize'] = None
    # params['num_samples'] = None
    # params['downsample_factor'] = 2
    # for _ in range(10):
    #     train_and_evaluate(params)
    #
    #
    #




