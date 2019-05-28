import random
import torch
import glob

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch.utils.data import DataLoader, random_split, RandomSampler, ConcatDataset, WeightedRandomSampler
from torch.utils.data import Dataset
from torch.nn import Dropout
from torchvision.transforms import Compose
from utilities.postprocessing.handlers import AddAdvancement, CleanDataframe, AddHMcoordinates, \
    open_df_and_hm_from_meta_row
from utilities.postprocessing.handlers.functional import add_advancement
from utilities.postprocessing.utils import hmpatch

random.seed(0)
np.random.seed(0)


class TraversabilityDataset(Dataset):
    def __init__(self, df, hm,
                 patches_dir,
                 patch_size=None,
                 tr=None,
                 time_window=None,
                 transform=None,
                 more_than=None,
                 less_than=None,
                 down_sampling=None,
                 transform_with_label=None,
                 ):

        self.df = df
        self.hm = hm
        self.patches_dir = patches_dir
        self.patch_size = patch_size
        self.tr = tr
        self.time_window = time_window

        self.transform = transform
        self.transform_with_label = transform_with_label
        self.should_generate_paths = not 'images' in df

        if 'advancement' not in self.df:
            self.df = add_advancement(self.df, time_window)

        if down_sampling is not None:
            self.df = self.df[::down_sampling]

        if more_than is not None: self.df = self.df[self.df['advancement'] >= more_than]
        if less_than is not None: self.df = self.df[self.df['advancement'] <= less_than]
        if tr is not None and len(self.df) > 0:
            self.df["label"] = self.df["advancement"] > tr

    def read_patch(self, img_name):
        patch = cv2.imread(self.patches_dir + '/' + img_name)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)
        patch /= 255
        return patch

    def generate_patch(self, row):
        patch = hmpatch(self.hm, row["hm_x"], row["hm_y"], np.rad2deg(row['pose__pose_e_orientation_z']),
                        self.patch_size,
                        scale=1)[0]

        return patch

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]

        if self.should_generate_paths:
            patch = self.generate_patch(row)
        else:
            patch = self.read_patch(row['images'])

        y = row['advancement']

        if 'label' in self.df:
            y = row['label'].astype(np.long)

        if 'height' in row:
            patch *= row['height']

        y = torch.tensor(y)

        if 'label' in self.df and self.transform_with_label is not None:
            patch = self.transform_with_label(patch, row['label'])

        return self.transform(patch), y

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_meta(cls, meta, base_dir, hm_dir, n=None, *args, **kwargs):
        datasets = []

        for (idx, row) in meta.iterrows():
            try:
                df, hm = open_df_and_hm_from_meta_row(row, base_dir, hm_dir)
            except FileNotFoundError:
                continue
            if len(df) > 0: datasets.append(cls(df, hm, *args, **kwargs))
        if n is not None: datasets = datasets[:n]
        concat_ds = ConcatDataset(datasets)
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'

        return concat_ds

    @staticmethod
    def concat_dfs(concat_ds):
        df = None
        for ds in concat_ds.datasets:
            if df is None:
                df = ds.df
            else:
                df = pd.concat([df, ds.df], sort=True)
        df = df.reset_index(drop=True)
        concat_ds.df = df
        return concat_ds

    @classmethod
    def from_root(cls, root, n=None, *args, **kwargs):
        dfs_paths = glob.glob(root + '/*.csv')
        if len(dfs_paths) == 0: dfs_paths = glob.glob(root + '/**/*.csv')
        datasets = []
        for df_path in dfs_paths:
            df = pd.read_csv(df_path)
            if len(df) > 0:
                datasets.append(cls(df, root, *args, **kwargs))
        if n is not None: datasets = datasets[:n]

        concat_ds = ConcatDataset(datasets)
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'

        return concat_ds

    @classmethod
    def from_dfs(cls, dfs, root, *args, **kwargs):
        datasets = []

        for df in dfs:
            if len(df) > 0:
                datasets.append(cls(df, root, *args, **kwargs))

        concat_ds = ConcatDataset(datasets)
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'
        return concat_ds


class PatchesDataset(Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
        self.df = pd.DataFrame()

    def __getitem__(self, item):
        patch = self.patches[item].hm
        if self.transform is not None: patch = self.transform(patch)

        return patch.float(), 0

    def __len__(self):
        return len(self.patches)

#
