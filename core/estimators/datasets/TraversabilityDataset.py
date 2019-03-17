import random
import torch
import glob

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, random_split, RandomSampler, ConcatDataset, WeightedRandomSampler
from torchvision.transforms import Resize, ToPILImage, ToTensor, Grayscale, Compose
from torch.utils.data import Dataset
from torch.nn import Dropout


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
                            iaa.CoarseDropout((0.02, 0.05),
                                              size_percent=(0.3, 0.5))

                        ], random_order=True)
                    )


def random_scale(x):
    random_scale = np.random.choice(np.arange(1, 5, 0.5))
    x *= random_scale
    return x, random_scale


class CenterAndScalePatch():
    """
    This class is used to center in the middle and rescale a given
    patch. We need to center in the  middle in order to
    decouple the root position from the classification task. Also,
    depending on the map, we need to multiply the patch by a scaling factor.
    """

    def __init__(self, scale=1.0, debug=False, should_aug=False):
        self.scale = scale
        self.debug = debug
        self.should_aug = should_aug

    def show_heatmap(self, x, title, ax):
        ax.set_title(title)
        img_n = x
        sns.heatmap(img_n,
                    # vmin=0,
                    # annot=True,
                    # linewidths=.5,
                    ax=ax,
                    fmt='0.2f')

    def random_scale_aug(self, x, is_traversable):
        if not is_traversable:
            scale = 1
            if self.should_aug:
                if np.random.rand() > 0.8:
                    x, scale = random_scale(x)
                if self.debug:
                    ax = plt.subplot(2, 2, 4)
                    self.show_heatmap(x, 'scale aug scale={} class={}'.format(scale, y), ax)
                    plt.show()

    def __call__(self, x, debug=False):
        if self.debug: fig = plt.figure()
        # is_traversable = y == 1
        if self.debug:
            ax = plt.subplot(2, 2, 1)
            self.show_heatmap(x, 'original', ax)

        x = x.astype(np.double)
        x = x / 255
        x *= self.scale

        center = x[x.shape[0] // 2, x.shape[1] // 2]
        x -= center

        min, max = x.min(), x.max()

        if self.debug:
            print(max, min)
            ax = plt.subplot(2, 2, 2)
            self.show_heatmap(x, 'centered', ax)

        if self.should_aug:
            x = (x - min) / (max - min)  # norm to 0,1 -> imgaug does not accept neg values
            x = aug.augment_image(x)
            x = x * (max - min) + min  # go back
        if self.debug:
            ax = plt.subplot(2, 2, 3)
            self.show_heatmap(x, 'aug', ax)

        if self.debug: plt.show()

        return x.astype(np.float32)


class TraversabilityDataset(Dataset):
    def __init__(self, df, transform, tr=None, more_than=None, should_aug=False, debug=False, downsample_factor=None):
        self.df = pd.read_csv(df)
        self.df = self.df.dropna()  # to be sure
        if downsample_factor is not None: self.df = self.df[0:-1:downsample_factor]
        if more_than is not None:  self.df = self.df[self.df['advancement'] >= more_than]
        self.transform = transform
        self.tr = tr
        self.idx2class = {'False': 0,
                          'True': 1}

        self.should_aug = should_aug
        self.debug = debug

    def __getitem__(self, item):
        row = self.df.iloc[item]
        img_path = row['image_path']
        # img = Image.open(img_path)
        y = row['advancement']
        y = torch.tensor(y)

        if self.tr is not None: y = 1 if y >= self.tr else 0

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return self.transform(img), y

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_root(cls, root, n=None, *args, **kwargs):
        dfs = glob.glob(root + '/*-patch.csv')
        if len(dfs) == 0: dfs = glob.glob(root + '/**/*-patch.csv')
        if n is not None: dfs = dfs[:n]
        print(root + '/**/*-patch.csv')
        concat_ds = ConcatDataset([cls(df, *args, **kwargs) for df in dfs])
        # needed for fastAI
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'

        return concat_ds


class FastAIImageFolder(TraversabilityDataset):
    c = 2
    classes = 'False', 'True'


def get_transform(resize, should_aug=False, scale=1, debug=False):
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
    transformations.append(CenterAndScalePatch(scale=scale, debug=debug, should_aug=should_aug))
    # if should_aug: transformations.append(ImgaugWrapper(aug, debug))
    transformations.append(ToTensor())
    # if should_aug: transformations.append(Dropout(0.1))

    return Compose(transformations)


def get_dataloaders(train_root, test_root, val_root=None,
                    val_size=0.2, tr=0.45,
                    num_samples=None, train_transform=None,
                    val_transform=None, test_transform=None,
                    more_than=None, should_aug=False,
                    downsample_factor=None,
                    *args,
                    **kwargs):
    """
    Get train, val and test dataloader.
    :return: train, val and test dataloaders
    """
    print('tr={}'.format(tr))
    print(train_transform, val_transform, test_transform)
    train_ds = FastAIImageFolder.from_root(root=train_root,
                                           transform=train_transform, tr=tr,
                                           should_aug=should_aug,
                                           downsample_factor=downsample_factor,
                                           more_than=more_than)

    train_size = int(len(train_ds) * (1 - val_size))

    if val_root is None:
        train_ds, val_ds = random_split(train_ds, [train_size, len(train_ds) - train_size])

    else:
        val_ds = FastAIImageFolder.from_root(root=val_root,
                                             transform=val_transform, tr=tr,
                                             more_than=more_than)

    if num_samples is not None:
        print('sampling')
        train_dl = DataLoader(train_ds,
                              sampler=RandomSampler(train_ds, num_samples=num_samples, replacement=True),
                              *args, **kwargs)
    else:
        train_dl = DataLoader(train_ds,
                              shuffle=True,
                              # sampler=ImbalancedDatasetSampler(train_ds),
                              *args, **kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, *args, **kwargs)

    test_ds = FastAIImageFolder.from_root(root=test_root,
                                          transform=test_transform,
                                          tr=tr,
                                          more_than=more_than)

    test_dl = DataLoader(test_ds, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl


def visualise(dl, n=10):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for (x, y) in dl:
        for i, img in zip(range(5), x):
            sns.heatmap(img.numpy().squeeze(), ax=axes[i], vmin=0, vmax=1)

        plt.show()
        break


if __name__ == '__main__':
    # df = '/home/francesco/Desktop/querry-high/df/querry-big-10/1552309429.462741-patch.csv'
    #
    df = '/home/francesco/Desktop/data/750/train/df/slope_rocks3/1550606526.7238998-patch.csv'
    ds = TraversabilityDataset(df, transform=get_transform(None, True, scale=10, debug=True), debug=True, tr=0.45)

    print(len(ds))
    # for i in  range(2):
    img, y = ds[0]
    print(y)

    # from torch.nn import Dropout
    # img = Dropout(0.1)(img)

    # train_dl, val_dl, test_dl = get_dataloaders(
    #     train_root='/home/francesco/Desktop/bars1-run-recorded/csvs-light/',
    #     test_root='/home/francesco/Desktop/data/92/test/',
    #     val_root='/home/francesco/Desktop/data/92/val',
    #     train_transform=get_transform(92, should_aug=True),
    #     val_transform=get_transform(92),
    #     test_transform=get_transform(92, scale=10),
    #     batch_size=5,
    #     num_samples=None,
    #     num_workers=1,
    #     pin_memory=True)
    #
    # #
    # visualise(train_dl)
    # visualise(train_dl)
    # visualise(train_dl)
    # visualise(train_dl)
    #
    # visualise(val_dl)
    # visualise(val_dl)

    # visualise(test_dl)
    # visualise(test_dl)
    # visualise(test_dl)
    # visualise(test_dl)
#
