import torch
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from os import path

from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from fastai.callback import Callback
from fastai.train import LearnerCallback


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

class StoreResults(Callback):
    """
    Store each input, target and prediction into a Dataframe in order to
    perform custom queries on the dataset.
    """

    def __init__(self, memory_size=30,):
        self.memory_size = memory_size
        self.save_dir = '/tmp/inputs.csv'

    def on_epoch_begin(self, **kwargs):
        self.df = None

    def on_batch_end(self, last_input, last_output, last_target, train, **kwargs):
        if not train:
            to_np = lambda x: x.cpu().numpy()
            last_target = to_np(last_target)
            last_output_act = softmax(last_output, dim=1)
            prediction = to_np(torch.argmax(last_output_act, dim=1))
            last_output_act = to_np(last_output_act)

            output_0, output_1 = last_output_act[:,0].tolist(), last_output_act[:,1].tolist()

            df = pd.DataFrame(data={'input': to_np(last_input).tolist(),
                                    'output_0': output_0,
                                    'output_1': output_1,
                                    'prediction': prediction.tolist(),
                                    'target': last_target.tolist() })

            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df])

            self.free_memory()

    @property
    def exist(self):
        return path.isfile(self.save_dir)

    def plot(self, sample):
        for img, pred, target, out0, out1 in zip(sample['input'], sample['prediction'], sample['target'], sample['output_0'], sample['output_1']):
            img = np.array(img).squeeze()
            fig = plt.figure()
            plt.title('pred={}, target={} ouputs=[{:.2f}, {:.2f}]'.format(pred, target, out0, out1))
            sns.heatmap(img)
            plt.show()

    def free_memory(self):
        """
        This function reduces the number of rows in the dataframe.
        If we store everything we will run out of RAM!
        :return:
        """
        n = len(self.df) if self.memory_size > len(self.df) else  self.memory_size
        print(len(self.df), n)
        self.df = self.df.sample(self.memory_size)

class StoreBestWorstAndSample(StoreResults):

    @property
    def best(self):
        return self.df.sort_values(['output_1'], ascending=False)

    @property
    def worst(self):
        return self.df.sort_values(['output_0'], ascending=False)

    def false_something(self, something):
        neg = self.df.loc[self.df['label'] == something]
        return neg.loc[neg['prediction'] != something]

    @property
    def false_pos(self):
        return self.false_something(0)

    @property
    def false_neg(self):
        return self.false_something(1)



