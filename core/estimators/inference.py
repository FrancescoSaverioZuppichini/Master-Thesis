import torch
import cv2

import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows

from models.omar_cnn import *
from fastai.train import Learner, DataBunch
from fastai.layers import CrossEntropyFlat
from datasets.TraversabilityDataset import get_transform


class InferenceDataset(Dataset):
    def __init__(self, hm_path, patch_size=100, step=1, transform=None):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
        # self.hm = self.hm.astype(np.float32)
        self.images = view_as_windows(self.hm, (patch_size, patch_size), step).reshape(-1, patch_size, patch_size)
        self.transform = transform

    def __getitem__(self, item):
        img = Image.fromarray(self.images[item])
        if self.transform: img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.images)


ds = InferenceDataset('/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png',
                      step=5,
                      transform=get_transform(64, scale=10))
dl = DataLoader(ds, batch_size=128, num_workers=16, shuffle=False)


data = DataBunch(train_dl=dl, valid_dl=dl, test_dl=dl)

model = OmarCNN()
criterion = CrossEntropyFlat()

learner = Learner(data=data,
                  model=model)

learner.load('/home/francesco/Desktop/carino/vaevictis/data/omar-100-100-0.1-12-no_tail-spawn-0.001-64-accuracy')

outs = learner.get_preds(data.test_dl)

_, preds = torch.max(outs[0], 1)

print(preds[0:100])
# loss, acc = learner.validate(data.valid_dl, metrics=[accuracy])