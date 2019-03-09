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


import skimage
import skimage.io
import skimage.feature
import skimage.novice
import skimage.transform


def hmpatch(hm, x, y, alpha, edge, scale=1):
    """
    Cutout a patch from the image, centered on (x,y), rotated by alpha
    degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
    with a specified edge size (in pixels) and scale (relative).
    :param hm:
    :param x:
    :param y:
    :param alpha: TODO ask
    :param edge: TODO ask
    :param scale: TODO ask
    :return:
    """
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge / 2, +edge / 2])
    tf = (tf1 + (tf2 + (tf3 + tf4))).inverse

    corners = tf(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * edge)
    patch = skimage.transform.warp(hm, tf, output_shape=(edge, edge), mode="edge")
    return patch, corners


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

        cv2.imwrite('/home/francesco/Desktop/textures/{}-{}.png'.format(name, self.rotate), texture)


ds = InferenceDataset('/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png',
                      patch_size=92,
                      step=3,
                      transform=get_transform(92, scale=10), rotate=0)

dl = DataLoader(ds, batch_size=128, num_workers=16, shuffle=False)

data = DataBunch(train_dl=dl, valid_dl=dl, test_dl=dl)

# model = OmarCNN()
model = MicroResnet.micro(1,
                          n=2,
                          blocks=[BasicBlock, BasicBlock, BasicBlock, BasicBlockSE],
                          preactivate=False)

criterion = CrossEntropyFlat()

learner = Learner(data=data,
                  model=model)

learner.load('/home/francesco/Desktop/carino/vaevictis/data/microresnet#4-gate=3x3-n=2-se=True-750-0.001-92-1552129741.4400764/roc_auc')

outs = learner.get_preds(data.test_dl)

_, preds = torch.max(outs[0], 1)

ds.make_texture(preds.numpy(), 'querry')
# ds.visualise(preds.numpy())
# ds.make_texture(preds.numpy(), 'bars1')
