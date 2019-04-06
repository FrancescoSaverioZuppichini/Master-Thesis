import tqdm
import os
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from mirror.visualisations.core import GradCam
from estimators.models import *
from estimators.utils import get_learner, get_probs_and_labels_from_preds, get_patches_form_df, load_model_from_name, \
    device
from estimators.datasets.TraversabilityDataset import TraversabilityDataset, get_transform

from os import path
from Config import Config
from utilities.patches import *

from utilities.postprocessing.postprocessing import Handler, Compose


class StorePredictions(Handler):
    def __init__(self, model_name, model_dir, store_dir, *args, **kwargs):
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
            if type(dataset) is not TraversabilityDataset: raise ValueError(
                'inputs must be of type TraversabilityDataset')
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


class FilterPatches(Handler):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = None
        self.transform = transform

    def transform_patches(self, patches):
        return [self.transform(patch) for patch in patches]

    def filter_patches(self, df, image_dir):
        return self.transform_patches(get_patches_form_df(df, image_dir))

    def handle(self, df, image_dir, filter_fn, n=2):
        filtered_df = filter_fn(df)
        filtered_df = filtered_df.head(n)
        return filtered_df, self.filter_patches(filtered_df, image_dir),

class Convert2Patches():
    def __call__(self, data):
        df, patches, grad_cams = data
        patches = Patch.from_tensors(patches)

        for (patch, grad_cam) in zip(patches, grad_cams):
            patch.add_texture(grad_cam)

        return df, patches

class GradCamVisualization(Handler):
    def __init__(self, model, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def handle(self, data):
        df, patches = data

        grad_cams = [self.get_grad_cam(patch) for patch in patches]

        return df, patches, grad_cams


class PlotInterestingPatches(Handler):
    def make_title(self, metric, row):
        return '{} advancement={:.2f} prediction = {} ground truth = {}'.format(metric, row['advancement'],
                                                                                row['prediction'],
                                                                                row['label'])

    def handle(self, data):
        for metric, (df, patches) in data.items():
            for (idx, row), patch in zip(df.iterrows(), patches):
                patch.plot2d(self.make_title(metric, row))
                patch.texture.plot2d()



class GetInterestingPatches(Handler):
    def __init__(self, transform, image_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        self.image_dir = image_dir
        self.filters = [Best(), Worst(), FalseNegative(), FalsePositive()]

    def handle(self, data):
        df = pd.concat(data)
        result = {}


        f_patch = FilterPatches(transform=transform)
        c_patch = Convert2Patches()
        g_patch  = GradCamVisualization(load_model_from_name(Config.BEST_MODEL_DIR + '/roc_auc.pth', Config.BEST_MODEL_NAME), device)

        filter_convert_and_grad_cam = Compose([f_patch, g_patch, c_patch])

        for f in self.filters:
            result[f.name] = filter_convert_and_grad_cam(df, self.image_dir, f)

        return result


transform = get_transform(scale=1)

concat = TraversabilityDataset.from_paths(Config.DATA_ROOT, [Config.DATA_DIR], tr=0.45, transform=transform)

plot_patches = PlotInterestingPatches()
get_patches = GetInterestingPatches(transform, Config.DATA_ROOT)
store_prediction = StorePredictions(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR,
                                    '/home/francesco/Desktop/store-test/')

pip = Compose([store_prediction, get_patches, plot_patches])
pip(concat.datasets)
