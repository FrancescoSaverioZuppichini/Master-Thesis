import torch
import random

import imgaug as ia
import numpy as np

from torch.nn import Module
from torch.utils.data import DataLoader
from datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from models import zoo
from fastai.train import Learner, DataBunch, DatasetType
from torch.nn.functional import softmax

def load_model(path: str, model: Module):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])

    return model


def get_learner(model_name, model_dir, callbacks, load_metric='roc_auc', dataset=None, *args, **kwargs):
    model = zoo[model_name]
    if dataset is None: dataset = TraversabilityDataset.from_root(*args, **kwargs)
    test_dl = DataLoader(dataset,
                         shuffle=False, batch_size=128, num_workers=8)

    learner = Learner(data=DataBunch(test_dl, test_dl, test_dl=test_dl), model=model,
                      callbacks=callbacks,
                      model_dir=model_dir)

    learner.load(load_metric)

    return learner


def load_model_from_name(model_path, model_name):
    model = zoo[model_name]
    return load_model(model_path, model)


def seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    ia.seed(seed)


def get_probs_and_labels_from_preds(preds):
    probs, _ = preds
    _, labels = torch.max(probs, 1)
    probs = softmax(probs, dim=1)

    return probs, labels


