import torch
import cv2
import imutils

import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
from skimage.util.shape import view_as_windows

class InferenceDataset(Dataset):
    """
    This class creates a dataset from an heightmap that can be used during inference
    to test the model.
    """
    def __init__(self, hm_path, patch_size=92, step=1, transform=None, rotate=None, debug=False):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)

        # rotate the heightmap and them cup the images = to cut the images and then rotate them
        # if rotate is not None: self.hm = imutils.rotate(self.hm, rotate)
        self.images = view_as_windows(self.hm, (patch_size, patch_size), step)
        self.images_shape = self.images.shape
        self.images = self.images.reshape(-1, patch_size, patch_size)
        self.transform = transform
        self.step = step
        self.patch_size = patch_size
        self.rotate = rotate
        self.debug = debug
        print(self.images_shape)

    def show_patch(self, patch, title):
        fig = plt.figure()
        plt.title(title)
        sns.heatmap(patch.squeeze())
        plt.show()

    def __getitem__(self, item):
        img = self.images[item]
        if self.debug: self.show_patch(img, 'original')

        if self.rotate is not None: img = imutils.rotate(img, self.rotate)
        if self.transform is not None: img = self.transform(img)

        if self.debug: self.show_patch(img, 'transform')

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

    def make_texture(self, outputs, predictions, name):
        texture = np.zeros(self.hm.shape)

        fig = plt.figure()
        sns.heatmap(imutils.rotate(self.hm, self.rotate))
        plt.title('rotation = {}'.format(self.rotate))
        plt.show()


        predictions = predictions.reshape(self.images_shape[0], self.images_shape[1])
        outputs = outputs.reshape(self.images_shape[0], self.images_shape[1], 2)

        w, h = self.hm.shape
        j = 0

        pbar = tqdm.tqdm(total=self.images.shape[0])

        for x in range(0, w, self.step):
            i = 0
            for y in range(self.step, h, self.step):
                try:
                    pred = predictions[i, j]
                    out = outputs[i,j]
                    is_traversable = pred == 1
                    # TODO understand why they are swapped
                    if is_traversable: texture[y:y + self.patch_size, x: x + self.patch_size] += out[1]

                    i += 1
                    pbar.update(1)
                except IndexError:
                    break
            j += 1

        pbar.close()

        texture = cv2.normalize(texture, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        texture = (texture * 255).astype(np.uint8)

        fig = plt.figure()
        sns.heatmap(texture)
        plt.show()

        path = '/Users/vaevictis/Documents/Project/Master-Thesis/resources/assets/textures/{}-{}.png'.format(name, self.rotate)

        cv2.imwrite(path, texture)

        return path


if __name__ == '__main__':
    from TraversabilityDataset import get_transform

    ds = InferenceDataset('../../maps/test/querry-big-10.png',
                          patch_size=92,
                          step=10,
                          transform=get_transform(None, scale=10), rotate=0, debug=True)

    ds[0]
    ds[1]

