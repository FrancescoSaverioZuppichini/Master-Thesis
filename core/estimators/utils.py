import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from models import zoo
from fastai.train import Learner, DataBunch, DatasetType

def load_model(path: str, model: Module):
    state = torch.load(path)
    model.load_state_dict(state['model'])


def get_learner(model_name, model_dir, callbacks, transform, load_metric='roc_auc', *args, **kwargs):
    model = zoo[model_name]
    test_dl = DataLoader(TraversabilityDataset.from_root(root='/home/francesco/Desktop/data/750/test/df/',
                                                         transform=transform, *args, **kwargs), shuffle=False, batch_size=128, num_workers=16)

    learner = Learner(data=DataBunch(test_dl, test_dl), model=model,
                      callbacks=callbacks,
                      model_dir='/home/francesco/Desktop/carino/vaevictis/data/{}/'.format(model_dir))

    learner.load(load_metric)

    return learner