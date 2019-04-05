import tqdm
import os
import glob
import torch

import pandas as pd
import cv2

from models import *
from utils import get_learner, get_probs_and_labels_from_preds, get_patches_form_df
from callbacks import StoreBestWorstAndSample, ROC_AUC
from os import path
from datasets.TraversabilityDataset import TraversabilityDataset, get_transform
from Config import Config

class StorePredictions():
    def __init__(self, model_name, model_dir, store_dir):
        self.model_name = model_name
        self.model_dir = model_dir
        self.dfs = []
        self.df_path2df = {}
        self.store_dir = store_dir

    def handle_dataset(self, dataset):
        df = dataset.df
        learner, _ = get_learner(self.model_name, self.model_dir, callbacks=[], dataset=dataset)
        preds = learner.get_preds(learner.data.test_dl)
        probs, labels = get_probs_and_labels_from_preds(preds)

        df['out_0'] = probs[:, 0]
        df['out_1'] = probs[:, 1]
        df['prediction'] = labels.cpu().tolist()

        return df

    def store(self):
        for df_path, df in tqdm.tqdm(self.df_path2df.items()):
            file_name = path.basename(df_path)
            map_name = path.basename(path.split(df_path)[0])
            out_path = path.normpath('{}/{}/'.format(self.store_dir, map_name))
            os.makedirs(out_path, exist_ok=True)
            out_path = path.normpath('{}/{}'.format(out_path, file_name))
            df.to_csv(out_path)

    def restore(self):
        dfs_path = glob.glob(self.store_dir + '/**/*.csv')

        for df_path in tqdm.tqdm(dfs_path):
            df = pd.read_csv(df_path)
            self.df_path2df[df_path] = df
            self.dfs.append(df)

    def __call__(self, datasets):
        bar = tqdm.tqdm(datasets)
        for dataset in bar:
            if type(dataset) is not TraversabilityDataset: raise ValueError('inputs must be of type TraversabilityDataset')
            bar.set_description('[INFO] Reading {}'.format(dataset.df_path))
            if len(dataset) > 0:
                df = self.handle_dataset(dataset)
                self.dfs.append(df)
                self.df_path2df[dataset.df_path] = self.dfs[-1]

        return self.dfs


class GetInterestingPatches():
    def __init__(self, transform=None):
        self.df = None
        self.transform = None

    @property
    def best(self):
        df = self.df.loc[self.df['label'] == 1]
        return df.sort_values(['out_1'], ascending=False)

    @property
    def worst(self):
        df = self.df.loc[self.df['label'] == 0]
        return df.sort_values(['out_0'], ascending=False)

    def false_something(self, something):
        neg = self.df.loc[self.df['label'] == something]
        return neg.loc[neg['prediction'] != something]

    @property
    def false_pos(self):
        return self.false_something(0)

    @property
    def false_neg(self):
        return self.false_something(1)

    def filter_patches(self, df, image_dir):
        return (df, get_patches_form_df(df, image_dir))

    def __call__(self, dfs, image_dir):
        df = pd.concat(dfs)

        return  { 'false_pos':  self.filter_patches(self.false_pos, image_dir),
                  'false_neg' : self.filter_patches(self.false_neg, image_dir),
                  'best': self.filter_patches(self.best, image_dir),
                  'worst': self.filter_patches(self.worst, image_dir)
                  }


concat = TraversabilityDataset.from_root(Config.DATA_ROOT, tr=0.45, transform=get_transform(scale=10))
store = StorePredictions(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR, '/home/francesco/Desktop/store-test/')

store.restore()

inter_patches = GetInterestingPatches(transform=get_transform(scale=10))
inter_patches([store.dfs[0]], Config.DATA_ROOT)

hm = cv2.imread('/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png')
hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

# store.restore()
# print(df['out_0'], df['out_1'], df['prediction'])





