from comet_ml import Experiment

import torch
import pprint
# from torchvision.models import *

from torchsummary import summary

from fastai.train import Learner, DataBunch, \
    ReduceLROnPlateauCallback, \
    EarlyStoppingCallback, \
    SaveModelCallback, DatasetType

from fastai.vision import ClassificationInterpretation

from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders, get_transform

from models.resnet import *
from models.omar_cnn import OmarCNN

from models.custom_resnet import *
import matplotlib.pyplot as plt

from functools import partial
import time
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)


def train_and_evaluate(params, train=True, load_model=None):
    # model = OmarCNN()
    model = MicroResnet.micro(1,
                              n_classes=2,
                              block=[BasicBlock, BasicBlock, BasicBlock, BasicBlockSE],
                              preactivated=True)
    # print(model)


    summary(model.cuda(), (1, params['resize'], params['resize']))
    pprint.pprint(params)
    criterion = CrossEntropyFlat()

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/home/francesco/Desktop/data/train/dataset/{}'.format(params['dataset']),
        test_root='/home/francesco/Desktop/data/test/dataset/{}'.format(params['test_dataset']),
        val_root='/home/francesco/Desktop/data/test/dataset/{}'.format(params['test_dataset']),
        # val_size=0.15,
        train_transform=get_transform(params['resize'], should_aug=params['data-aug']),
        val_transform=get_transform(params['resize'], scale=10),
        test_transform=get_transform(params['resize'], scale=10),
        num_samples=params['sampler'],
        batch_size=params['batch_size'],
        num_workers=16,
        pin_memory=True)

    print("train size={}, val size={}, test size={}".format(
        len(train_dl) * params['batch_size'],
        len(val_dl) * params['batch_size'],
        len(test_dl) * params['batch_size']
    ))

    data = DataBunch(train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl)


    experiment = Experiment(api_key="8THqoAxomFyzBgzkStlY95MOf",
                            project_name="master-thesis",
                            workspace="francescosaveriozuppichini")

    experiment.log_parameters(params)

    learner = Learner(data=data,
                      model=model,
                      path='/home/francesco/Desktop/carino/vaevictis/data/',
                      model_dir='/home/francesco/Desktop/carino/vaevictis/data/',
                      loss_func=criterion,
                      opt_func= partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      metrics=[accuracy])

    model_name_acc = '{}-{}-{}-{}-accuracy-{}'.format(params['model'], params['dataset'], params['lr'], params['resize'], time.time())
    model_name_loss = '{}-{}-{}-{}-loss-{}'.format(params['model'], params['dataset'], params['lr'], params['resize'], time.time())

    callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=4),
                 EarlyStoppingCallback(learn=learner, patience=6),
                 SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'),
                 SaveModelCallback(learn=learner, name=model_name_loss)]

    if train:
        try:
            with experiment.train():
                learner.fit(epochs=params['epochs'], lr=params['lr'],
                            callbacks=callbacks)  # SaveModelCallback load the best model after training!
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

    learner.load(model_name_acc)

    torch.save(learner.model, '/home/francesco/Desktop/carino/vaevictis/data/{}.pck'.format(params['model']))

    if load_model:
        print('loading model')
        learner.load(load_model)

    with experiment.test():
        loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
        print(loss, acc)
        experiment.log_metric("accuracy-from-best-acc", acc.item())


    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix(normalize=True, title='Val')
    plt.savefig(learner.model_dir + '/' + model_name_acc + '.png')
    # experiment.log_image('/home/francesco/Desktop/carino/vaevictis/data/' + load_model + '-valid.png')
    plt.show()

    interp = ClassificationInterpretation.from_learner(learner, ds_type=DatasetType.Test)
    interp.plot_confusion_matrix(normalize=True, title='Test')
    plt.savefig(learner.model_dir + '/' + model_name_acc + '.png')
    # experiment.log_image('/home/francesco/Desktop/carino/vaevictis/data/' + load_model + '-test.png')
    plt.show()

params = {'epochs': 50,
          'lr': 0.001,
          'batch_size': 128,
          # 'model': '128omar',
          'model': 'microresnet#3-preactivate=True-se=True-gate=5x5-2-pool-2-1',
          'dataset': '100-92-0.08-25-no_tail-spawn-shift#2-no-neg',
          'val_dataset': '100-92-0.08-12-no_tail-spawn-shift-no-neg',
          'test_dataset': '100-92-0.08-12-querry-no_tail-spawn-shift-no-neg',
          'sampler': None,
          'samper_type': 'imbalance',
          'callbacks': '[ReduceLROnPlateauCallback]',
          'data-aug': True,
          'optim': 'sdg',
          'info': 'test as val',
          'resize': 92}

# train_and_evaluate(params, train=False, load_model='microresnet#3-preactivate=True-se=True-100-92-0.12-25-no_tail-spawn-shift#2-0.001-92-accuracy-True')
train_and_evaluate(params, train=True)
# params['data-aug'] = True
# train_and_evaluate(params, train=True)
# params['sampler'] = True
# train_and_evaluate(params, train=True)


# params['data-aug'] = False
# train(params)
