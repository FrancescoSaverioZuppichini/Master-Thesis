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
from estimators.data.transformations import RandomSimplexNoise, DropoutAgumentation, RandomScale
from estimators.data.dataloaders import ImbalancedDatasetSampler

from torch.utils.data.sampler import RandomSampler

from utilities.postprocessing.utils import KrockPatchExtractStrategy

torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt


class TrainAndEvaluate():
    def __init__(self, params):
        self.params = params
        self.train_dl, self.val_dl, self.test_dl = get_dataloaders(
            train_root='/media/francesco/saetta/krock-dataset/train/',
            hm_root='/home/francesco/Documents/Master-Thesis/core/maps/new-train/',
            patch_size=params['patch_size'],
            test_root=params['test'],
            test_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/test/',
            val_root=params['validation'],
            # val_root='/media/francesco/saetta/krock-dataset/val/',
            val_hm_root='/home/francesco/Documents/Master-Thesis/core/maps/val/' if params[
                                                                                        'validation'] is not None else None,
            tr=params['tr'],
            time_window=params['time_window'],
            train_transform=params['train_transform'],
            train_transform_with_label=params['train_transform_with_label'],
            val_transform=params['val_transform'],
            test_transform=params['test_transform'],
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

    def __call__(self, model):
        params = self.params
        # model = OmarCNN()
        if type(model) is str:
            model = zoo[model]()
        else:
            model = model()
        patch_size = KrockPatchExtractStrategy.patch_shape(params['patch_size'], 0.02)
        summary(model.cuda(), (1, patch_size[0], patch_size[1]))
        # print(model)
        pprint.pprint(params)
        timestamp = time.time()

        criterion = CrossEntropyFlat() if params['tr'] is not None else MSELossFlat()

        experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                                project_name="i-love-krock", workspace="francescosaveriozuppichini")

        experiment.log_parameters(params)
        experiment.log_metric("timestamp", timestamp)

        model_name = '{}-{}-{}-{}-{}'.format(params['model'], params['dataset'].split('/')[0], params['lr'],
                                             params['patch_size'], timestamp)

        model_dir = path.normpath('/media/francesco/Carino/vaevictis/data/' + model_name + '/')

        os.makedirs(model_dir, exist_ok=True)

        print("train size={}, val size={}, test size={}".format(
            len(self.train_dl) * params['batch_size'],
            len(self.val_dl) * params['batch_size'],
            len(self.test_dl) * params['batch_size']
        ))

        data = DataBunch(train_dl=self.train_dl, valid_dl=self.val_dl, test_dl=self.test_dl)

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
            ReduceLROnPlateauCallback(learn=learner, patience=4, factor=0.2),
            EarlyStoppingCallback(learn=learner, patience=8),
            CSVLogger(learn=learner),
            # SaveModelCallback(learn=learner, name=model_name_loss)
        ]

        if params['tr'] is not None:
            callbacks.append(SaveModelCallback(learn=learner, name=model_name_roc_auc, monitor='roc_auc'))
            # callbacks.append(SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'))
            # callbacks.append(SaveModelCallback(learn=learner, name=model_name_loss))

        with experiment.train():
            # learner.lr_find()
            # learner.recorder.plot()
            # plt.show() # 1e-01
            # lr = 1e-3
            learner.fit_one_cycle(5, slice(params['lr']), pct_start=0.8, callbacks=callbacks)
            # # lr =  1e-4,
            # learner.fit_one_cycle(5, slice(params['lr']), pct_start=0.8, callbacks=callbacks)
            # loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            # print(loss, acc, roc)
            #
            # # learner.lr_find()
            # # learner.recorder.plot()
            # # plt.show() # 1e
            learner.fit_one_cycle(5, slice(1e-5), callbacks=callbacks)
            # loss, acc, roc = learner.validate(data.test_dl, metrics=[accuracy, ROC_AUC()])
            # print(loss, acc, roc)
            # # #
            # learner.fit_one_cycle(5, slice(1e-6), pct_start=0.8, callbacks=callbacks)

            #
            # learner.fit(epochs=params['epochs'], lr=params['lr'],
            #             callbacks=callbacks)  # SaveModelCallback load the best model after training!

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

        del learner.model
        del learner
        del model


def get_params():
    return {'epochs': 30,
            'lr': 1e-3,
            'batch_size': 128,
            'val_size': 10,
            'test': None,
            'validation': None,
            'model': 'microresnet#4-gate=3x3-n=1-se=False',
            'dataset': '',
            'sampler': None,
            'num_samples': None,
            'data-aug': (None, None),
            'train_transform_with_label': None,
            'train_transform': get_transform(),
            'val_transform': get_transform(),
            'test_transform': get_transform(),
            # 'data-aug': None,
            # 'optim': partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
            'optim':  torch.optim.Adam,
            'info': 'diobestia',
            'tr': 0.2,
            'problem': 'classification',
            'name': '',
            'more_than': None,
            'less_than': None,
            'down_sampling': None,
            'time_window': 50 * 2,
            'patch_size': 0.71
            }


if __name__ == '__main__':
    from estimators.models.resnet import *
    from torchvision.transforms import ToTensor, Compose
    from estimators.data.transformations import RandomCoarsening, RandomSimplexNoise, CenterAndScalePatch, \
        DropoutAgumentation
    from torch.nn import Dropout

    class MyDecoder(ResnetDecoder):
        def __init__(self, in_features, n_classes):
            super().__init__(in_features, n_classes)
            #         self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.decoder = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.Dropout(),
                nn.Linear(64, n_classes),
            )


    class Encoder3x3(ResNetEncoder):
        def __init__(self, in_channel, depths, *args, **kwargs):
            super().__init__(in_channel, depths, *args, **kwargs)

            self.gate = nn.Sequential(
                nn.Conv2d(in_channel, self.blocks_sizes[0][0], kernel_size=3, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(self.blocks_sizes[0][0]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )


    class Encoder7x7(ResNetEncoder):
        def __init__(self, in_channel, depths, *args, **kwargs):
            super().__init__(in_channel, depths, *args, **kwargs)

            self.gate = nn.Sequential(
                nn.Conv2d(in_channel, self.blocks_sizes[0][0], kernel_size=7, stride=2, bias=False, padding=2),
                nn.BatchNorm2d(self.blocks_sizes[0][0]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )


    class Encoder5x5(ResNetEncoder):
        def __init__(self, in_channel, depths, *args, **kwargs):
            super().__init__(in_channel, depths, *args, **kwargs)

            self.gate = nn.Sequential(
                nn.Conv2d(in_channel, self.blocks_sizes[0][0], kernel_size=5, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(self.blocks_sizes[0][0]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )


    validate_transformation = Compose([CenterAndScalePatch(), ToTensor])
    train_transformation = Compose([
        # RandomCoarsening(p=0.6),
        CenterAndScalePatch(),
        # DropoutAgumentation(),
        ToTensor(),
        DropoutAgumentation(),
    ])

    # r_scale = RandomScale(p=0.3)
    # r_noise = RandomSimplexNoise(p=0.7, n=500)
    # def train_transform_with_label(img, is_traversable):
    #     x = r_scale(img, is_traversable)
    #     x = r_noise(x, is_traversable)
    #     return x


    params = get_params()
    params['train_transform'] = train_transformation
    params['train_transform_with_label'] = RandomSimplexNoise(p=0.7, n=500)
    params['data-aug'] = str(train_transformation)
    patch_size = KrockPatchExtractStrategy.patch_shape(params['patch_size'], 0.02)

    params['test'] = '/media/francesco/saetta/krock-dataset/new-test-random/'
    # params['validation'] = '/media/francesco/saetta/krock-dataset/new-new-val/'
    # params['validation'] = '/media/francesco/saetta/krock-dataset/new-test-random/'
    # params['sampler'] = RandomSampler
    # params['num_samples'] = 50000

    model = lambda: ResNet(
        in_channel=1,
        encoder=Encoder7x7,
        # decoder=MyDecoder,
        depths=[1, 1, 1],
        blocks=[BasicBlockSE, BasicBlockSE, BasicBlockSE],
        blocks_sizes=[(16, 32), (32, 64), (64, 128)],
        n_classes=2,
        activation='leaky_relu',
        preactivate=True,
        ratio=4,
    )

    # model()
    params['model'] = model

    train_and_evaluate = TrainAndEvaluate(params)
    for _ in range(5):
        train_and_evaluate( params['model'])
