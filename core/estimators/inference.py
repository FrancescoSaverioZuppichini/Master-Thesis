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

from models.omar_cnn import *
from fastai.train import Learner, DataBunch
from fastai.layers import CrossEntropyFlat
from datasets.TraversabilityDataset import get_transform
from models.custom_resnet import *
from fastai.vision import ClassificationInterpretation


class InferenceDataset(Dataset):
    def __init__(self, hm_path, patch_size=80, step=1, transform=None, rotate=None):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
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
        img = Image.fromarray(self.images[item])
        # img.show()
        if self.rotate is not None: img = rotate(img, self.rotate)
        # img.show()
        if self.transform: img = self.transform(img)

        return img, 0

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

                    func(x,y, is_traversable)
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
                                                  self.patch_size, linewidth=0, edgecolor='none', facecolor='b', alpha=0.1)
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

        cv2.imwrite('/home/francesco/Desktop/textures/{}-{}.png'.format(name, self.rotate), texture)


ds = InferenceDataset('/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png',
                      patch_size=92,
                      step=3,
                      transform=get_transform(92, scale=1), rotate=None)

dl = DataLoader(ds, batch_size=128, num_workers=16, shuffle=False)

data = DataBunch(train_dl=dl, valid_dl=dl, test_dl=dl)

# model = OmarCNN()
model = MicroResnet.micro(1,
                          n_classes=2,
                          block=[BasicBlock, BasicBlock, BasicBlock, BasicBlockSE],
                          preactivated=True)
criterion = CrossEntropyFlat()

learner = Learner(data=data,
                  model=model)

learner.load(
    '/home/francesco/Desktop/carino/vaevictis/data/microresnet#3-preactivate=True-se=True-gate=5x5-2-pool-2-1-100-92-0.06-25-no_tail-spawn-shift#2-0.001-92-accuracy-1551544152.1612258')

outs = learner.get_preds(data.test_dl)

_, preds = torch.max(outs[0], 1)

ds.make_texture(preds.numpy(), 'bars1')
# ds.visualise(preds.numpy())

# ds.make_texture(preds.numpy(), 'bars1')
