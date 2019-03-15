import torch
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rotate
from skimage.util.shape import view_as_windows

from models import zoo
from fastai.train import Learner, DataBunch
from fastai.layers import CrossEntropyFlat
from datasets.TraversabilityDataset import get_transform
from models.custom_resnet import *
from utils import get_learner
import cv2
import imutils

class InferenceDataset(Dataset):
    def __init__(self, hm_path, patch_size=92, step=1, transform=None, rotate=None):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
        if rotate is not None:
            self.hm = imutils.rotate(self.hm, rotate)
        # self.hm = self.hm.astype(np.float32)
        self.images = view_as_windows(self.hm, (patch_size, patch_size), step)
        self.images_shape = self.images.shape
        self.images = self.images.reshape(-1, patch_size, patch_size)


        self.transform = transform
        self.step = step
        self.patch_size = patch_size
        self.rotate = rotate

        print(self.images_shape)

    def __getitem__(self, item):
        img = self.images[item]
        # img.show()
        # img.show()
        if self.transform is not None:
            img = self.transform(img)

        # if self.rotate is not None:
        #     img = imutils.rotate(img, self.rotate)

        return img, torch.tensor(0)

    def __len__(self):
        return len(self.images)

    def iter_patches(self, predictions, func):
        w, h = self.hm.shape

        j = 0
        for x in range(0, w, self.step):
            i = 0
            for y in range(self.step, h, self.step):
                try:
                    pred = predictions[i, j]
                    is_traversable = pred == 0

                    func(x, y, is_traversable)
                    i += 1
                except IndexError:
                    break
        j += 1

    def add_rects(self, predictions, ax):

        w, h = self.hm.shape

        j = 0
        for x in range(0, w, self.step):
            i = 0
            for y in range(self.step, h, self.step):
                try:
                    pred = predictions[i, j]
                    is_traversable = pred == 0

                    if is_traversable == 0:
                        rect = mpatches.Rectangle((x, y), self.patch_size,
                                                  self.patch_size, linewidth=0, edgecolor='none', facecolor='b',
                                                  alpha=0.1)
                        ax.add_patch(rect)

                    i += 1
                except IndexError:
                    break

            j += 1

    def visualise(self, predictions):
        fig = plt.figure()
        plt.title('rotation={}'.format(self.rotate))
        ax = plt.subplot(1, 1, 1)

        sns.heatmap(self.hm, ax=ax)

        predictions = predictions.reshape(self.images_shape[0], self.images_shape[1])

        self.add_rects(predictions, ax)

        plt.show()

    def make_texture(self, predictions, name):
        texture = np.zeros(self.hm.shape)
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        predictions = predictions.reshape(self.images_shape[0], self.images_shape[1])

        w, h = self.hm.shape
        buffer = []
        j = 0
        for x in range(0, w, self.step):
            i = 0
            for y in range(self.step, h, self.step):
                try:
                    pred = predictions[i, j]
                    is_traversable = pred == 1
                    # TODO understand why they are swapped
                    if is_traversable: texture[y:y + self.patch_size, x: x + self.patch_size] += 1

                    i += 1
                except IndexError:
                    break
            j += 1

        sns.heatmap(texture)

        path = '/home/francesco/Desktop/textures/{}-{}.png'.format(name, self.rotate)
        if self.rotate is not None:
            texture= imutils.rotate(texture, -self.rotate)
        cv2.imwrite(path, texture)

        return path

def run_inference(rotate=0):
    ds = InferenceDataset('/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png',
                          patch_size=92,
                          step=4,
                          transform=get_transform(None, scale=10), rotate=rotate)


    model_dir = 'microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294'
    # model_dir = 'microresnet#4-gate=3x3-n=2-se=True-750-0.001-92-1552226514.811947'

    model_name = 'microresnet#4-gate=3x3-n=2-se=True'
    model = zoo[model_name]

    learner = get_learner(model_name, model_dir, callbacks=[], dataset=ds)

    outs = learner.get_preds(learner.data.test_dl)

    _, preds = torch.max(outs[0], 1)

    ds.make_texture(preds.numpy(), 'querry-best')

for angle in [0, 90, 180, 270]:
    run_inference(angle)