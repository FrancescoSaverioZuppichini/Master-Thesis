import torch
import random

import imgaug as ia
import numpy as np

from torch.nn import Module
from torch.utils.data import DataLoader
from datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from models import zoo
from fastai.train import Learner, DataBunch, DatasetType

def load_model(path: str, model: Module):
    state = torch.load(path)
    model.load_state_dict(state['model'])


def get_learner(model_name, model_dir, callbacks, load_metric='roc_auc', dataset=None, *args, **kwargs):
    model = zoo[model_name]
    if dataset is None: dataset = TraversabilityDataset.from_root(*args, **kwargs),
    test_dl = DataLoader(dataset,
                         shuffle=False, batch_size=128, num_workers=16)

    learner = Learner(data=DataBunch(test_dl, test_dl, test_dl=test_dl), model=model,
                      callbacks=callbacks,
                      model_dir='/home/francesco/Desktop/carino/vaevictis/data/{}/'.format(model_dir))

    learner.load(load_metric)

    return learner


def seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    ia.seed(seed)