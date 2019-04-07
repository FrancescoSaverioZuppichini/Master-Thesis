import torch
import random

import imgaug as ia
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from torch.nn import Module
from torch.utils.data import DataLoader
from estimators.datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from estimators.models import zoo
from fastai.train import Learner, DataBunch, DatasetType
from torch.nn.functional import softmax

def load_model(path: str, model: Module):
    state = torch.load(path)
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

    return learner, dataset

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

def false_something(self, something):
    correct = self.df.loc[self.df['label'] == something]
    return correct.loc[correct['prediction'] != something]

def hmshow(hm, title='', *args, **kwargs):
    fig = plt.figure()
    plt.title(title)
    sns.heatmap(hm.squeeze(), *args, **kwargs)
    plt.show()
    return fig

def get_patches_form_df(df, image_dir):
    patches = []

    if 'images' in df.columns: patches = df['images']
    else:
        for img_path in df['image_path']:
            img = cv2.imread(image_dir + img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            patches.append(img)

    return patches



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')