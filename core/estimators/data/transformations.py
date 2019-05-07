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

class ImgaugWrapper():
    """
    Wrapper for imgaug
    """

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x):
        x = self.aug.augment_image(x)
        return x

simplex = OpenSimplex()
def im2simplex(im, feature_size=24, scale=10):
    h, w = im.shape[0], im.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            value = simplex.noise2d(x / feature_size, y / feature_size)
            im[x,y] += value / scale
    return im

class RandomSimplexNoise(Augmenter):

    def __init__(self, shape=(93, 93), n=10, *args, **kwargs):
        super().__init__()
        self.images = []
        self.n = n
        image = np.zeros((shape))

        for _ in tqdm(range(self.n)):
            features_size = np.random.randint(1, 50)
            im = im2simplex(image.copy(), features_size, 1)
            im = np.expand_dims(im, -1)
            self.images.append(im)

    def augment_image(self, img,   *args, **kwargs):
        idx = np.random.randint(0, self.n)
        scale = np.random.randint(4, 8)

        return img + (self.images[idx] / scale)

        # features_size = np.random.randint(15, 80)
        # scale = np.random.randint(5, 10)
        # return im2simplex(img, features_size, scale)

    def _augment_images(self, images, *args, **kwargs):
        for i in range(len(images)):
            images[i] = self.augment_image(images[i], *args, **kwargs)

        return images


    def _augment_heatmaps(self, *args, **kwargs):
        return None

    def _augment_polygons(self, *args, **kwargs):
        return None

    def get_parameters(self):
        return None

    def _augment_keypoints(self, *args, **kwargs):
        return None
# TODO this must be in a zoo and be passed as param to CenterAndScalePatch

def get_aug():
    return  iaa.Sometimes(0.8,
                    iaa.Sequential(
                        [
                            iaa.Dropout(p=(0.05, 0.1)),
                            iaa.CoarseDropout((0.02, 0.1),
                                              size_percent=(0.6, 0.8)),
                            RandomSimplexNoise(n=500)

                        ], random_order=False),

                    )

class CenterAndScalePatch():
    """
    This class is used to center in the middle and rescale a given
    patch. We need to center in the  middle in order to
    decouple the root position from the classification task. Also,
    depending on the map, we need to multiply the patch by a scaling factor.
    """

    def __init__(self, scale=1.0, debug=False, should_aug=False, resize=None, aug=None):
        self.scale = scale
        self.debug = debug
        self.should_aug = should_aug
        self.resize = resize
        self.aug = None
        if self.should_aug: self.aug = get_aug() if aug is None else aug

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

        min, max = x.min() + 1e-5, x.max()

        if self.debug:
            ax = plt.subplot(2, 2, 3)
            self.show_heatmap(x, 'centered {}'.format(center), ax)

        if self.should_aug:
            x = (x - min) / (max - min)  # norm to 0,1 -> imgaug does not accept neg values
            x = self.aug.augment_image(x)
            x = x * (max - min) + min  # go back

        if self.resize is not None:
            x = cv2.resize(x, self.resize)
        if self.debug:
            ax = plt.subplot(2, 2, 4)
            self.show_heatmap(x, 'final', ax)

        if self.debug: plt.show()
        return x.astype(np.float32)


def get_transform(should_aug=False, scale=1, debug=False, resize=None):
    """
    Return a `Compose` transformation to be applied to the input of the model
    :param resize: size in pixel of the wanted final patch size
    :param should_aug: if True, dropout will be applied on the input
    :param scale: integer that is multiplied to the input
    :return:
    """
    transformations = []
    # if resize is not None: transformations.append(Resize((resize, resize)))
    # transformations.append(ToTensor())
    transformations.append(CenterAndScalePatch(scale=scale, debug=debug, should_aug=should_aug, resize=resize))
    # if should_aug: transformations.append(ImgaugWrapper(aug, debug))
    transformations.append(ToTensor())
    # if should_aug: transformations.append(Dropout(0.1))

    return Compose(transformations)