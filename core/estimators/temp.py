import tqdm
import os
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from mirror.visualisations.core import GradCam
from estimators.models import *
from estimators.utils import get_learner, get_probs_and_labels_from_preds, read_patch, get_patches_form_df, load_model_from_name, \
    device
from estimators.datasets.TraversabilityDataset import TraversabilityDataset, get_transform, PatchesDataset

from os import path
from Config import Config
from utilities.patches import *

from utilities.postprocessing.postprocessing import Handler, Compose


class StorePredictionsHandler(Handler):
    def __init__(self, model_name, model_dir, store_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def handle(self, datasets):
        bar = tqdm.tqdm(datasets)
        for dataset in bar:
            # if type(dataset) is not Data: raise ValueError(
            #     'inputs must be of type PatchesDataset')
            if hasattr(dataset, 'df_patch'): bar.set_description('[INFO] Reading {}'.format(dataset.df_path))
            if len(dataset) > 0:
                df = self.handle_dataset(dataset)
                self.dfs.append(df)
                if isinstance(dataset, PatchesDataset): df['images'] = dataset.patches
                if hasattr(dataset, 'df_patch'): self.df_path2df[dataset.df_path] = self.dfs[-1]

        return pd.concat(self.dfs)


def false_something(df, something):
    neg = df.loc[df['label'] == something]
    return neg.loc[neg['prediction'] != something]



class Best():
    name = 'best'

    def __call__(self, df):
        df = df.loc[df['prediction'] == 1]

        return df.sort_values(['out_1'], ascending=False)

class Worst():
    name = 'worst'

    def __call__(self, df):
        df = df.loc[df['prediction'] == 0]

        return df.sort_values(['out_0'], ascending=False)


class FalseNegative():
    name = 'false_negative'

    def __call__(self, df):
        return false_something(df, 0)


class FalsePositive():
    name = 'false_positive'

    def __call__(self, df):
        return false_something(df, 1)


class FilterDataframeHandler(Handler):
    def __init__(self, pip_filtered=None, filters=None, n=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.pip_filtered = pip_filtered
        self.n = n

    def handle(self, df):
        if self.filters is None: return { 'all' : df}
        res = { f.name : self.pip_filtered(f(df).head(self.n)) for f in self.filters}
        return res

class GetPatchesFromDataframeHandler(Handler):
    def __init__(self, image_dir=None, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dir = image_dir
        self.transform = transform

    def handle(self, df):
        if 'images' not in df.columns:  df['images'] =  [Patch.from_tensor(transform(read_patch(self.image_dir + img_path))) for img_path in df['image_path']]
        return df

class PlotPatchesFromDataframe(Handler):
    def __init__(self, title='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title

    def plot_df(self, df, title=''):
        for idx, row in df.iterrows():
            patch = row['images']
            patch.plot3d('[{}] Prediction = {} Confidence = [{:.2f} {:.2f}] Between = {}'.format(title,
                row['prediction'],
                row['out_0'],
                row['out_1'],
                "")) #TODO __repr__ can be too long!

    def plot_dict(self, dict):
        for metric, df in dict.items():
            self.plot_df(df, metric)

    def handle(self, data):
        if type(data) is dict: self.plot_dict(data)
        else: self.plot_df(data)


transform = get_transform(scale=1, debug=False)

get_patches = GetPatchesFromDataframeHandler(image_dir=Config.DATA_ROOT, transform=transform)

run_model_on_dataset = Compose([StorePredictionsHandler(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR),
                                FilterDataframeHandler(filters=[Best(), Worst()])])

concat = TraversabilityDataset.from_paths(Config.DATA_ROOT, [Config.DATA_DIR], tr=0.45, transform=transform)


get_and_plot_df = Compose([GetPatchesFromDataframeHandler(image_dir=Config.DATA_ROOT), PlotPatchesFromDataframe()])

run_model_on_custom_patches = Compose([StorePredictionsHandler(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR),
                                       FilterDataframeHandler(filters=[Best(), Worst()], pip_filtered=get_patches),
                                       PlotPatchesFromDataframe()],
                                      )


patches = BarPatch.from_range(shape=(88,88), strength=0.3, offset=list(range(20, 44, 4)))
ds = PatchesDataset(patches, transform=transform)

run_model_on_custom_patches([ds])
# res = run_model_on_dataset([concat.datasets[0]])

# print(res)