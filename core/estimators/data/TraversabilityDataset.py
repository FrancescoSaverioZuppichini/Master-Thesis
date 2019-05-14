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
from utilities.postprocessing.postprocessing import AddAdvancement, CleanDataframe, AddHMcoordinates, \
    open_df_and_hm_from_meta_row
from utilities.postprocessing.utils import hmpatch

random.seed(0)
np.random.seed(0)


class TraversabilityDataset(Dataset):
    def __init__(self, df, hm,
                 patches_dir,
                 time_window,
                 patch_size,
                 tr=None,
                 transform=None,
                 more_than=None,
                 down_sampling=None,
                 simplex_noise=None):
        self.hm = hm
        self.patches_dir = patches_dir
        self.patch_size = patch_size
        self.tr = tr
        self.transform = transform
        self.df = df
        self.simplex_noise = simplex_noise
        self.should_generate_paths = not 'images' in df

        self.preprocess_df = Compose([AddAdvancement(time_window)])
        self.df = self.preprocess_df((self.df, None, None))[0]

        if down_sampling is not None:
            self.df = self.df[::down_sampling]


        if more_than is not None: self.df = self.df[self.df['advancement'] >= more_than]
        if tr is not None: self.df["label"] = (self.df["advancement"] > tr)

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
            patch = cv2.imread(self.patches_dir + '/' + row['images'])
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            patch = patch.astype(np.float32)
            patch /= 255

        y = row['advancement']

        if 'label' in self.df:
            y = row['label'].astype(np.long)

        if 'height' in row:
            patch *= row['height']

        y = torch.tensor(y)

        if 'label' in self.df and  self.simplex_noise is not None:
            if np.random.random() > 0.2:
                patch = self.simplex_noise(patch, row['label'])


        patch =  self.transform(patch)


        return patch, y

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_meta(cls, meta, base_dir, hm_dir,  n=None, *args, **kwargs):
        datasets = []

        for (idx, row) in meta.iterrows():
            df, hm = open_df_and_hm_from_meta_row(row, base_dir, hm_dir)
            if len(df) > 0: datasets.append(cls(df, hm, *args, **kwargs))

        if n is not None: datasets = datasets[:n]
        concat_ds = ConcatDataset(datasets)
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'
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

        return patch, 0

    def __len__(self):
        return len(self.patches)




#
