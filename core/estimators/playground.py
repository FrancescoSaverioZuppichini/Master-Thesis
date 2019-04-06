import tqdm
import os
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from mirror.visualisations.core import GradCam
from models import *
from utils import get_learner, get_probs_and_labels_from_preds, get_patches_form_df, load_model_from_name, device
from callbacks import StoreBestWorstAndSample, ROC_AUC
from os import path
from datasets.TraversabilityDataset import TraversabilityDataset, get_transform
from Config import Config
from patches import *

from utils.postprocessing.postprocessing import Handler

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


def false_something(df, something):
    neg = df.loc[df['label'] == something]
    return neg.loc[neg['prediction'] != something]

class Best():
    name = 'best'
    def __call__(self, df):
        df = df.loc[df['label'] == 1]
        return df.sort_values(['out_1'], ascending=False)

class Worst():
    name = 'worst'
    def __call__(self, df):
        df = df.loc[df['label'] == 0]
        return df.sort_values(['out_0'], ascending=False)

class FalseNegative():
    name = 'false_negative'
    def __call__(self, df):
        return false_something(df, 0)

class FalsePositive():
    name = 'false_positive'
    def __call__(self, df):
        return false_something(df, 1)


class FilterPatches():
    def __init__(self, transform=None):
        self.df = None
        self.transform = transform

    def transform_patches(self, patches):
        return [self.transform(patch) for patch in patches]

    def filter_patches(self, df, image_dir):
        return self.transform_patches(get_patches_form_df(df, image_dir))

    def __call__(self, df, image_dir, filter_fn, n=2):
        filtered_df = filter_fn(df)
        filtered_df = filtered_df.head(n)
        return filtered_df, self.filter_patches(filtered_df, image_dir),


class Convert2Patches():
    def __call__(self, data):
        df, patches = data
        return df, Patch.from_tensors(patches)

class GradCamVisualization():

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.grad_cam = GradCam(model.to(self.device), self.device)

    def get_grad_cam(self, patch):
        img = patch.unsqueeze(0).to(self.device)

        _, info = self.grad_cam(img, None, target_class=None)

        cam = info['cam'].cpu().numpy()
        cam = cv2.resize(cam, (patch.shape[1], patch.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def __call__(self, data):
        df, patches = data

        grad_cams = [self.get_grad_cam(patch) for patch in patches]

        return df, patches, grad_cams


class PlotInterestingPatches():
    def plot_all(self, patches, title):
        for patch in patches:
            patch.plot3d(title)

    def make_title(self, metric, row):
        return '{} advancement={:.2f} prediction = {} ground truth = {}'.format(metric, row['advancement'], row['prediction'],
                                                                             row['label'])

    def __call__(self, data):
        for metric, (df, patches) in data.items():
            for (idx, row), patch in zip(df.iterrows(), patches):
                patch.plot3d(self.make_title(metric, row))

def get_all_interesting_patches(transform, df, image_dir):
    filters = [Best(), Worst(), FalseNegative(), FalsePositive()]
    result = {}

    f_patch = FilterPatches(transform=transform)
    c_patch = Convert2Patches()

    for f in filters:
        result[f.name] = c_patch(f_patch(df, image_dir, f))

    return result

transform = get_transform(scale=1)

concat = TraversabilityDataset.from_paths(Config.DATA_ROOT, [Config.DATA_DIR], tr=0.45, transform=transform)
store = StorePredictions(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR, '/home/francesco/Desktop/store-test/')
dfs = store(concat.datasets)

data = get_all_interesting_patches(transform, dfs[0], Config.DATA_ROOT)

plot_patches = PlotInterestingPatches()
plot_patches(data)



