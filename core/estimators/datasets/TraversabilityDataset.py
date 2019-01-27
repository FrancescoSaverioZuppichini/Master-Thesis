import torch
import cv2
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

from utils.postprocessing.utils import files2dfs_maps
from utils.postprocessing.config import Config
import pandas as pd
import numpy as np
from os import path
import glob

class TraversabilityDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.img_dir =path.dirname(csv_path) + '/images'

    def __getitem__(self, item):
        print('{}/{}.png'.format(self.img_dir, self.df['name'][item]))
        input = cv2.imread('{}/{}.png'.format(self.img_dir, self.df['name'][item]))
        target = self.df['label'][item]

        return input, target

    def __len__(self):
        return len(self.df)


csv_paths = glob.glob(Config.IMAGES_DATASET_FOLDER + '/**/**/*.csv')
print(csv_paths[0])
ds = TraversabilityDataset(csv_paths[0])
print(len(ds))
print(ds[0])