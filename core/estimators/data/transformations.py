import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path
from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
from torchvision.transforms import Resize, ToPILImage, ToTensor, Grayscale, Compose
from opensimplex import OpenSimplex
from tqdm import tqdm


class DropoutAgumentation():
    """
    Wrapper for imgaug
    """

    def __init__(self):
        self.aug_prob = 0.8
        self.p = (0.05, 0.1)
        self.aug = iaa.Sometimes(self.aug_prob,
                                 iaa.Sequential(
                                     [
                                         iaa.Dropout(p=self.p),
                                         iaa.CoarseDropout(self.p,
                                                           size_percent=(0.6, 0.8))

                                     ], random_order=False),

                                 )

    def __call__(self, x):
        x = self.aug.augment_image(x)
        min, max = x.min() + 1e-5, x.max()

        x = (x - min) / (max - min)  # norm to 0,1 -> imgaug does not accept neg values
        x = self.aug.augment_image(x)
        x = x * (max - min) + min  # go back
        return x

    def __str__(self):
        return 'DropoutAgumentation={}p={}-({},size_percent=(0.6, 0.8))'.format(self.aug_prob, self.p, self.p)


simplex = OpenSimplex()


def im2simplex(im, feature_size=24, scale=10):
    h, w = im.shape[0], im.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            value = simplex.noise2d(x / feature_size, y / feature_size)
            im[x, y] += value / scale
    return im


class RandomSimplexNoise():
    """
    Apply random simplex noise on input based on p. This class is
    specific to our problem with hard-coded features dimension for each
    class.
    """

    def __init__(self, shape=(78, 78), n=10, p=0.8, *args, **kwargs):
        super().__init__()
        self.images = []
        self.n = n
        self.p = p
        self.features_dims = (3, 50)
        self.traversable_scale_dims = (15, 25)
        self.no_traversable_scale_dims = (3, 15)

        self._make_images(shape)

    def _make_images(self, shape):
        image = np.zeros((shape))

        for _ in tqdm(range(self.n)):
            features_size = np.random.randint(*self.features_dims)
            im = im2simplex(image.copy(), features_size, 1)
            self.images.append(im)

    def __call__(self, img, is_traversable):
        if np.random.random() > (1.0 - self.p):

            idx = np.random.randint(0, self.n)

            if is_traversable:
                scale = np.random.randint(*self.traversable_scale_dims)
            else:
                scale = np.random.randint(*self.no_traversable_scale_dims)

            tmp = self.images[idx] / scale
            img = img + tmp

        return img

    def __str__(self):
        return 'RandomSimplexNoise={}{}-{}-{}'.format(self.p, self.features_dims,
                                                      self.traversable_scale_dims,
                                                      self.no_traversable_scale_dims)


class CenterAndScalePatch():
    """
    This class is used to center in the middle and rescale a given
    patch. We need to center in the  middle in order to
    decouple the root position from the classification task. Also,
    depending on the map, we need to multiply the patch by a scaling factor.
    """

    def __init__(self, scale=1.0, debug=False, resize=None, ):
        self.scale = scale
        self.debug = debug
        self.resize = resize

    def show_heatmap(self, x, title, ax):
        ax.set_title(title)
        img_n = x
        sns.heatmap(img_n,
                    ax=ax,
                    fmt='0.2f')

    def __call__(self, x, debug=False):
        if self.debug: fig = plt.figure()

        if self.debug:
            ax = plt.subplot(2, 2, 1)
            self.show_heatmap(x, 'original', ax)

        x *= self.scale
        if self.debug:
            ax = plt.subplot(2, 2, 2)
            self.show_heatmap(x, 'scale', ax)

        center = x[x.shape[0] // 2, x.shape[1] // 2]
        x -= center

        if self.debug:
            ax = plt.subplot(2, 2, 3)
            self.show_heatmap(x, 'centered {}'.format(center), ax)
        if self.resize is not None:
            x = cv2.resize(x, self.resize)
        if self.debug:
            ax = plt.subplot(2, 2, 4)
            self.show_heatmap(x, 'final', ax)

        if self.debug: plt.show()
        return x.astype(np.float32)


def get_transform(aug=None, scale=1, debug=False, resize=None):
    """
     Return a `Compose` transformation to be applied to the input of the model
    :param scale: integer that is multiplied to the input
    :param aug: data augmentation
    :param resize: size in pixel of the wanted final patch size
    :return:ImbalancedDatasetSampler
    """
    transformations = []
    transformations.append(CenterAndScalePatch(scale=scale, debug=debug, resize=resize))
    if aug is not None: transformations.append(aug)
    transformations.append(ToTensor())

    return Compose(transformations)
