# from postprocessing import *
import os
import os
import glob
import cv2
import dateutil
import rosbag_pandas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import path
from utilities.postprocessing.utils import *
from pypeln import thread as th
from utilities.pipeline import Compose, Handler

class PostProcessingConfig():
    def __init__(self, maps_folder, patch_size, advancement_th, time_window, skip_every, translation,
                 resolution=0.02, scale=1, n_workers=16,
                 base_dir=None, csv_dir=None, out_dir=None, patch_dir=None, verbose=True, patches=True, name=''):
        self.maps_folder, self.base_dir, self.out_dir, self.csv_dir = maps_folder, base_dir, out_dir, csv_dir

        self.bags_dir = self.base_dir + '/bags/'

        self.patch_size, self.advancement_th, self.time_window = patch_size, advancement_th, time_window
        self.scale, self.skip_every = scale, skip_every
        self.resolution, self.translation = resolution, translation
        self.n_workers = n_workers
        self.name = name
        if out_dir is None: self.out_dir = base_dir

        self.out_dir = os.path.normpath(self.out_dir + name)

    @property
    def dataset_name(self):
        return '/{}'.format(self.patch_size)

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))


class MultiThreadWrapper(Handler):
    def __init__(self, pip, n_workers):
        self.n_workers = n_workers
        self.pip = pip

    def __call__(self, data):
        return list(tqdm(th.map(self.pip, data, workers=self.n_workers), total=len(data)))


class PostProcessingHandler(Handler):
    def __init__(self, config: PostProcessingConfig):
        self.config = config


def make_path(file_path, out_dir):
    splitted = file_path.split('/')
    map_name, file_name = splitted[-2], splitted[-1]
    return path.normpath(
        '{}/{}/{}'.format(out_dir + '/csvs/', map_name, path.splitext(file_name)[0]))


class Bags2Dataframe(Handler):
    """
    This class loads the bags file and converted them to Pandas' dataframe. In addition,
    it opens each map for each file and add it to the return tuple.
    """

    def __init__(self, base_dir, out_dir=None):
        self.base_dir = base_dir
        self.out_dir = out_dir if out_dir is not None else base_dir

    def __call__(self, file_name):
        df = rosbag_pandas.bag_to_dataframe(self.base_dir + file_name + '.bag')
        df.to_csv(self.out_dir + file_name + '.csv')
        return df

df = pd.read_csv('/media/francesco/saetta/KROCK-NEW-DATASET/train/meta.csv')
print(df)
base_dir = '/media/francesco/saetta/KROCK-NEW-DATASET/train/'
out_dir = '/media/francesco/saetta/KROCK-NEW-DATASET/train/csv/'

Compose([MultiThreadWrapper(Bags2Dataframe(base_dir=base_dir, out_dir=out_dir), n_workers=16)])(df['filename'])