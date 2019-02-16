from comet_ml import Experiment

import torch
from torchsummary import summary
from torch.utils.data import DataLoader, random_split, RandomSampler
from torchvision.datasets import ImageFolder

from fastai.train import Learner, DataBunch, load_learner
from fastai.metrics import accuracy

from datasets.TraversabilityDataset import get_transform

from models.resnet import *
from models.omar_cnn import OmarCNN

from models.custom_resnet import *

from functools import partial

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

test_root = '/home/francesco/Desktop/data/test/dataset/100-100-0.09-12-querry'
model_name = 'resnet34-100-100-0.09-12-06-02-19-0.0001-100-loss'

test_ds = ImageFolder(root=test_root,
                      transform=get_transform(100, 10))

test_dl = DataLoader(test_ds, batch_size=128, num_workers=16)

print(len(test_dl) * 128)

data = DataBunch(train_dl=test_dl, valid_dl=test_dl)

learner = load_learner('/home/francesco/Desktop/carino/vaevictis/data/{}'.format(model_name))



loss, acc = learner.validate(data.valid_dl, metrics=[accuracy])
print(loss, acc)