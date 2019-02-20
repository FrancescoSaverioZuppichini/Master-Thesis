from comet_ml import Experiment

import torch
import pprint
# from torchvision.models import *

from torchsummary import summary

from fastai.train import Learner, DataBunch, \
    ReduceLROnPlateauCallback, \
    EarlyStoppingCallback, \
    SaveModelCallback
from fastai.metrics import accuracy
from fastai.layers import CrossEntropyFlat

from datasets.TraversabilityDataset import get_dataloaders, get_transform

from models.resnet import *
from models.omar_cnn import OmarCNN

from models.custom_resnet import *
import matplotlib.pyplot as plt

from functools import partial

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)


def train(params):
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
        val_root='/home/francesco/Desktop/data/val/dataset/{}'.format(params['val_dataset']),
        # val_size=0.15,
        train_transform=get_transform(params['resize'], should_aug=params['data-aug']),
        val_transform=get_transform(params['resize']),
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
                      # opt_func= partial(torch.optim.SGD, momentum=0.95, weight_decay=1e-4),
                      metrics=[accuracy])

    model_name_acc = '{}-{}-{}-{}-accuracy'.format(params['model'], params['dataset'], params['lr'], params['resize'], params['data-aug'])
    model_name_loss = '{}-{}-{}-{}-loss'.format(params['model'], params['dataset'], params['lr'], params['resize'],  params['data-aug'])

    callbacks = [ReduceLROnPlateauCallback(learn=learner, patience=4),
                 EarlyStoppingCallback(learn=learner, patience=6),
                 SaveModelCallback(learn=learner, name=model_name_acc, monitor='accuracy'),
                 SaveModelCallback(learn=learner, name=model_name_loss)]
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

    with experiment.test():
        loss, acc = learner.validate(data.test_dl, metrics=[accuracy])
        print(loss, acc)
        experiment.log_metric("accuracy-from-best-acc", acc.item())



params = {'epochs': 50,
          'lr': 0.001,
          'batch_size': 128,
          'model': 'microresnet#3-preactive=True-se=True',
          'dataset': '100-92-0.12-25-no_tail-spawn-shift#2',
          'val_dataset': '100-92-0.12-12-no_tail-spawn-shift',
          'test_dataset': '100-92-0.12-12-querry-no_tail-spawn-shift',
          'sampler': None,
          'samper_type': 'sample',
          'callbacks': '[ReduceLROnPlateauCallback]',
          'data-aug': True,
          'optim': 'adam',
          'info': '',
          'resize': 92}

train(params)



# params['data-aug'] = False
# train(params)
