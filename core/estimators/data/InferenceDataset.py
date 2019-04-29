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
from PIL import Image
from estimators.data.utils import hm_patch_generator, hm_patch_list
from utilities.postprocessing.utils import KrockPatchExtractStrategyNumpy

class InferenceDataset(Dataset):
    """
    This class creates a dataset from an height map that can be used during inference
    to test the model.
    """
    def __init__(self, hm_path, patch_size=88, step=1, max_advancement=1, transform=None, rotate=None, debug=False, res=0.02):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
        self.hm_rot = imutils.rotate(self.hm, rotate)
        self.images = hm_patch_list(self.hm_rot, step, rotate, max_advancement=max_advancement, res=res)

        self.patch_size = (self.images.shape[2], self.images.shape[3])
        # self.images = view_as_windows(self.hm, (patch_size, patch_size), step)
        self.images_shape = (self.images.shape[0], self.images.shape[1])
        self.images = self.images.reshape(-1, *self.patch_size)

        self.transform = transform
        self.step = step
        self.patch_size = patch_size
        self.rotate = rotate
        self.debug = debug
        self.max_advancement = max_advancement

    def show_patch(self, patch, title):
        fig = plt.figure()
        plt.title(title)
        sns.heatmap(patch.squeeze())
        plt.show()

    def __getitem__(self, item):
        img = self.images[item]
        # img = self.images)

        if self.debug: self.show_patch(img, 'original')

        # if self.rotate is not None:
        #     img = np.array(Image.fromarray(img).rotate(self.rotate))

        img = img.astype(np.float32)
        img /= 255

        if self.transform is not None: img = self.transform(img)

        if self.debug: self.show_patch(img, 'transform')

        return img, torch.tensor(0)

    def __len__(self):
        return self.images_shape[0] * self.images_shape[1]

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
        hm_rot = np.array(Image.fromarray(self.hm).rotate(self.rotate))
        sns.heatmap(hm_rot)
        plt.title('rotation = {}'.format(self.rotate))
        plt.show()


        predictions = predictions.reshape(self.images_shape[0], self.images_shape[1])
        outputs = outputs.reshape(self.images_shape[0], self.images_shape[1], 2)

        w, h = self.hm.shape
        j = 0

        pbar = tqdm.tqdm(total=self.__len__())
        counter = np.zeros_like(texture)
        counter += 0.00001
        for x in range(0, w, self.step):
            i = 0
            for y in range(self.step, h, self.step):
                try:
                    pred = predictions[i, j]
                    out = outputs[i, j]
                    is_traversable = pred == 1
                    # TODO understand why they are swapped
                    if is_traversable:
                        texture = KrockPatchExtractStrategyNumpy.fill(texture, y, x, out[1], self.max_advancement)
                        # texture[y - self.patch_size[0] // 2 :y + self.patch_size[0] // 2,
                        #         x - self.patch_size[1]//2: x + self.patch_size[1] // 2] += out[1]
                    #
                    i += 1
                    pbar.update(1)
                except IndexError:
                    break
            j += 1

        pbar.close()

        texture[texture <= 0] = 0

        texture = cv2.normalize(texture, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        texture = imutils.rotate(texture, -self.rotate)

        fig = plt.figure()
        sns.heatmap(texture)
        plt.show()

        path = '/home/francesco/Documents/Master-Thesis/resources/assets/textures/{}-{}.png'.format(name, self.rotate)

        texture = (texture * 255).astype(np.uint8)
        cv2.imwrite(path, texture)

        return path


if __name__ == '__main__':
    ds = InferenceDataset('../../maps/test/querry-big-10.png', patch_size=(66, 76), rotate=0, debug=True, step=200, max_advancement=0.66)

    for i in range(30):
        ds[i]

