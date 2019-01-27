import torch
import cv2
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils.postprocessing.utils import files2dfs_maps
from utils.postprocessing.config import Config
import pandas as pd
import numpy as np
from os import path
import glob

ds = ImageFolder(root=Config.IMAGES_DATASET_FOLDER + '/images')
print(len(ds))
img, label = ds[0]

img.show()