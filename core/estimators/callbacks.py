import torch
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


class StoreBestWorstAndSample(Callback):
    """
    Store each input, target and prediction into a Dataframe in order to
    perform custom queries on the dataset.
    """

    def on_epoch_begin(self, **kwargs):
        self.df = None
        self.df_sample = None

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
                self.df_sample = df.sample(1)
            else:
                self.df = pd.concat([self.df, df])
                self.df_sample = pd.concat([self.df_sample, df.sample(1)])

            self.free_memory()

    def free_memory(self):
        """
        This function reduces the number of rows in the dataframe.
        If we store everything we will run out of RAM!
        :return:
        """

        best = self.df.sort_values(['output_1'], ascending=False).head(30)
        worst = self.df.sort_values(['output_0'], ascending=False).head(30)

        self.df = pd.concat([best, worst])


    def plot(self, sample):
        for img, pred in zip(sample['input'], sample['prediction']):
            img = np.array(img).squeeze()
            fig = plt.figure()
            plt.title(pred)
            sns.heatmap(img,
                        vmin=0,
                        # annot=True,
                        # linewidths=.5,
                        fmt='0.2f')
            plt.show()