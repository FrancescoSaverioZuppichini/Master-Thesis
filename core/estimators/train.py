from comet_ml import Experiment

import os
import torch
import pprint
import numpy as np
# from torchvision.models import *

from torch.nn.functional import softmax
from torchsummary import summary
from os import path
from fastai.train import Learner, DataBunch, \
    ReduceLROnPlateauCallback, \
    EarlyStoppingCallback, \
    SaveModelCallback, DatasetType

from fastai.vision import ClassificationInterpretation
from fastai.callback import Callback
from fastai.metrics import accuracy, dice
from fastai.layers import CrossEntropyFlat, MSELossFlat

from sklearn.metrics import roc_auc_score
from datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset

from models.resnet import *
from models.omar_cnn import OmarCNN

from models.custom_resnet import *
import matplotlib.pyplot as plt

from functools import partial
import time
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


def roc_auc(input, targs):

    preds = softmax(input, dim=1)
    targs = targs.cpu().numpy()
    hot_targs = np.zeros((targs.shape[0], 2))
    hot_targs[np.arange(targs.shape[0]), targs] = 1

    try:
        roc = roc_auc_score(hot_targs, preds.cpu().numpy())
    except ValueError:
        roc = 1
        pass
    return torch.tensor(roc)


def custom_accuracy(y_pred, y_true, thresh:float=0.01):
    # print(y_pred[0:10], y_true[0:10])
    distance = (y_pred - y_true).abs()
    acc = (distance < thresh).float().mean()
    return acc



if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = MicroResnet.micro(1,
                              n=5,
                              blocks=[BasicBlock, BasicBlock, BasicBlock, BasicBlock],
                              preactivate=False)
    # print(model)


    summary(model.cuda(), (1, params['resize'], params['resize']))
    pprint.pprint(params)

    criterion = CrossEntropyFlat()

    # criterion = MSELossFlat()

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/home/francesco/Desktop/data/{}/train/'.format(params['dataset']),
        test_root='/home/francesco/Desktop/data/{}/test/'.format(params['dataset']),
        val_root='/home/francesco/Desktop/data/{}/val'.format(params['dataset']),
        # val_size=0.15,
        train_transform=get_transform(params['resize'], should_aug=params['data-aug']),
        val_transform=get_transform(params['resize'], scale=1),
        test_transform=get_transform(params['resize'], scale=10),
        num_samples=params['num_samples'],
        batch_size=params['batch_size'],
        num_workers=16,
        tr=params['tr'],
        remove_negative=params['remove-negative'],
        pin_memory=True)

    model_name = '{}-{}-{}-{}-{}'.format(params['model'], params['dataset'].split('/')[0], params['lr'], params['resize'], time.time())
    model_dir = path.normpath('/home/francesco/Desktop/carino/vaevictis/data/' + model_name + '/')

    os.makedirs(model_dir)

    print("train size={}, val size={}, test size={}".format(
        len(train_dl) * params['batch_size'],
        len(val_dl) * params['batch_size'],
        len(test_dl) * params['batch_size']
    ))

    data = DataBunch(train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl)

    experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                            project_name="krock",
                            workspace="francescosaveriozuppichini")

    experiment.log_parameters(params)

    learner = Learner(data=data,
                      model=model,
                      path='/home/francesco/Desktop/carino/vaevictis/data/',
                      model_dir=model_dir,
                      loss_func=criterion,
                      opt_func= partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      metrics=[accuracy, roc_auc])



    model_name_acc = 'accuracy'
    model_name_loss = 'loss'

    callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=4),
                 EarlyStoppingCallback(learn=learner, patience=6),
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

    learner.load(model_name_loss)


    with experiment.test():
        loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
        print(loss, acc)
        experiment.log_metric("accuracy", acc.item())
        experiment.log_metric("test_loss", loss)

    learner.load(model_name_acc)


    if load_model:
        print('loading model')
        learner.load(load_model)

    with experiment.test():
        loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
        print(loss, acc)
        experiment.log_metric("accuracy-from-best-acc", acc.item())


    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix(normalize=True, title='Val')
    plt.savefig(learner.model_dir + '/' + 'val.png')
    plt.show()

    interp = ClassificationInterpretation.from_learner(learner, ds_type=DatasetType.Test)
    interp.plot_confusion_matrix(normalize=True, title='Test')
    plt.savefig(learner.model_dir + '/' + 'test.png')
    plt.show()

    print(model_name_acc)

params = {'epochs': 100,
          'lr': 0.001,
          'batch_size': 128,
          # 'model': 'omar',
          'model': 'microresnet#4-gate=7x7',
          'dataset': '92',
          'sampler': None,
          'num_samples': None,
          'samper_type': 'imbalance',
          'callbacks': '[ReduceLROnPlateauCallback]',
          'data-aug': False,
          'optim': 'sdg',
          'info': '',
          'tr' : 0.12,
          'remove-negative': False,
          'resize': 92}

# train_and_evaluate(params, train=False, load_model='microresnet#3-preactivate=True-se=True-100-92-0.12-25-no_tail-spawn-shift#2-0.001-92-accuracy-True')
# train_and_evaluate(params, train=True)
# params['resize'] = 64
# train_and_evaluate(params, train=True)
# params['resize'] = 92

params['data-aug'] = True
train_and_evaluate(params, train=True)
params['remove-negative'] = True
train_and_evaluate(params, train=True)
params['data-aug'] = False
train_and_evaluate(params, train=True)