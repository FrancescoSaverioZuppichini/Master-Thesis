import torch
import time

import numpy as np

from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

from fastai.callback import Callback


class ROC_AUC(Callback):
    """
    Compute roc auc by storing all the predictions/targets and run it once
    when the epoch is over
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = np.array([])
        self.targs = np.array([])

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            preds = softmax(last_output, dim=1)
            # TODO this can be done in one shot later
            preds = torch.argmax(preds, dim=1).long().cpu().numpy()
            targs = last_target.cpu().numpy()

            self.preds = np.append(self.preds, preds)
            self.targs = np.append(self.targs, targs)

    def on_epoch_end(self, last_metrics, last_target, last_output, **kwargs):
        roc = roc_auc_score(self.preds, self.targs)
        self.metric = roc
        self.preds = np.array([])
        self.targs = np.array([])
        return {'last_metrics': last_metrics + [torch.tensor(self.metric)]}


class Timer(Callback):
    def on_epoch_begin(self, **kwargs):
        self.start = time.time()

    def on_epoch_end(self, train, last_metrics, **kwargs):
        self.metric = time.time() - self.start

        return {'last_metrics': last_metrics + [self.metric]}