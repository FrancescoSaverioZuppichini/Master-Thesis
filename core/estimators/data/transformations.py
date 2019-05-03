import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path
from imgaug import augmenters as iaa
from torchvision.transforms import Resize, ToPILImage, ToTensor, Grayscale, Compose

class ImgaugWrapper():
    """
    Wrapper for imgaug
    """

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x):
        x = self.aug.augment_image(x)
        return x


aug = iaa.Sometimes(0.8,
                    iaa.Sequential(
                        [
                            iaa.Dropout(p=(0.05, 0.1)),
                            iaa.CoarseDropout((0.02, 0.1),
                                              size_percent=(0.4, 0.8))

                        ], random_order=True)
                    )


class CenterAndScalePatch():
    """
    This class is used to center in the middle and rescale a given
    patch. We need to center in the  middle in order to
    decouple the root position from the classification task. Also,
    depending on the map, we need to multiply the patch by a scaling factor.
    """

    def __init__(self, scale=1.0, debug=False, should_aug=False, resize=None):
        self.scale = scale
        self.debug = debug
        self.should_aug = should_aug
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

        min, max = x.min() + 1e-5, x.max()

        if self.debug:
            ax = plt.subplot(2, 2, 3)
            self.show_heatmap(x, 'centered {}'.format(center), ax)

        if self.should_aug:
            x = (x - min) / (max - min)  # norm to 0,1 -> imgaug does not accept neg values
            x = aug.augment_image(x)
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