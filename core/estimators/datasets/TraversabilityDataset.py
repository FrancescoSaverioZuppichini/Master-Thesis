import random
import torch
import glob

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torchvision

from os import path
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
                                              size_percent=(0.2 , 0.4))

                        ], random_order=True)
                    )




class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super().__init__(dataset)

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

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


        x = x.astype(np.double)
        x = x / 255

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
            print(max, min)
            ax = plt.subplot(2, 2, 3)
            self.show_heatmap(x, 'centered {}'.format(center), ax)

        if self.should_aug:
            x = (x - min) / (max - min)  # norm to 0,1 -> imgaug does not accept neg values
            x = aug.augment_image(x )
            x = x * (max - min) + min  # go back

        if self.resize is not None:
            x = cv2.resize(x, self.resize)
        if self.debug:
            ax = plt.subplot(2, 2, 4)
            self.show_heatmap(x, 'final', ax)

        if self.debug: plt.show()
        return  x.astype(np.float32)


class TraversabilityDataset(Dataset):
    def __init__(self, df, root, transform, tr=None, more_than=None, should_aug=False, debug=False, downsample_factor=None, only_forward=False):
        self.df = pd.read_csv(df)
        self.df = self.df.dropna()  # to be sure
        if downsample_factor is not None: self.df = self.df[::downsample_factor]
        if more_than is not None:  self.df = self.df[self.df['advancement'] >= more_than]
        self.transform = transform
        self.tr = tr
        self.image_dir =root
        self.idx2class = {'False': 0,
                          'True': 1}
        self.only_forward = only_forward

        self.should_aug = should_aug
        self.debug = debug

    def __getitem__(self, item):
        row = self.df.iloc[item]
        img_path = path.normpath(self.image_dir + row['image_path'])

        y = row['advancement']
        y = torch.tensor(y)

        if self.tr is not None: y = 1 if y >= self.tr else 0

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img = cv2.resize(img, (64,64))

        if self.only_forward: img = img[:, img.shape[-1] // 2: ]

        return self.transform(img), y

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_root(cls, root, n=None, *args, **kwargs):
        dfs = glob.glob(root + '/df/*.csv')
        if len(dfs) == 0:
            dfs = glob.glob(root + '/df/**/*.csv')
        if n is not None: dfs = dfs[:n]

        concat_ds = ConcatDataset([cls(df, root, *args, **kwargs) for df in dfs])
        # needed for fastAI
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'
        print(len(concat_ds))
        return concat_ds


class FastAIImageFolder(TraversabilityDataset):
    c = 2
    classes = 'False', 'True'


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


def get_dataloaders(train_root, test_root, val_root=None,
                    val_size=0.2, tr=0.45,
                    num_samples=None, train_transform=None,
                    val_transform=None, test_transform=None,
                    more_than=None, should_aug=False,
                    downsample_factor=None,
                    only_forward=False,
                    *args,
                    **kwargs):
    """
    Get train, val and test dataloader.
    :return: train, val and test dataloaders
    """
    print(train_transform, val_transform, test_transform)
    train_ds = FastAIImageFolder.from_root(root=train_root,
                                           transform=train_transform,
                                           tr=tr,
                                           should_aug=should_aug,
                                           downsample_factor=downsample_factor,
                                           more_than=more_than,
                                           only_forward=only_forward)

    train_size = int(len(train_ds) * (1 - val_size))

    if val_root is None:
        train_ds, val_ds = random_split(train_ds, [train_size, len(train_ds) - train_size])

    else:
        val_ds = FastAIImageFolder.from_root(root=val_root,
                                             transform=val_transform,
                                             tr=tr,
                                             more_than=more_than)

    if num_samples is not None:
        print('[INFO] Sampling')
        train_dl = DataLoader(train_ds,
                              # sampler=ImbalancedDatasetSampler(train_ds, num_samples=num_samples),

                              sampler=RandomSampler(train_ds, num_samples=num_samples, replacement=True),
                              *args, **kwargs)
    else:
        train_dl = DataLoader(train_ds,
                              shuffle=True,
                              *args, **kwargs)
    val_dl = DataLoader(val_ds, shuffle=True, *args, **kwargs)

    test_ds = FastAIImageFolder.from_root(root=test_root,
                                          transform=test_transform,
                                          tr=tr,
                                          more_than=more_than)

    test_dl = DataLoader(test_ds, shuffle=True, *args, **kwargs)

    return train_dl, val_dl, test_dl


def visualise(dl, n=10):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for (x, y) in dl:
        for i, img in zip(range(5), x):
            sns.heatmap(img.numpy().squeeze(), ax=axes[i], vmin=0, vmax=1)

        plt.show()
        break


if __name__ == '__main__':
    df = '/media/francesco/saetta/125-750/test/df/querry-big-10/1550308680.946694-complete.csv-patch.csv'
    df = '/media/francesco/saetta/125-750/train/df/bars1/1550616885.070434-complete.csv-patch.csv'
    from torch.nn.functional import softmax

    # root = path.abspath(
    #     '.. /../../resources/assets/datasets/test/')

    # root = '/media/francesco/saetta/125-750/test/'
    root = '/media/francesco/saetta/125-750/train/'

    # df = root + '/df/querry-big-10/1550307709.2522066-patch.csv'
    ds = TraversabilityDataset(df, root=root, transform=get_transform(resize=(64,64), should_aug=True, scale=10,
                                                                      debug=True),
                              tr=0.45, only_forward=False, downsample_factor=2)

    for i in  range(3):
        img, y = ds[i]
        plt.imshow(img.cpu().squeeze().numpy())
        plt.show()
        print(y)

    # from torch.nn import Dropout
    # img = Dropout(0.1)(img)
    dataset = '125-750'
    resize = None

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/media/francesco/saetta/{}/train/'.format(dataset),
        val_root='/media/francesco/saetta/{}/val/'.format(dataset),
        test_root='/media/francesco/saetta/{}/test/'.format(dataset),
        train_transform=get_transform(resize=resize, should_aug=True),
        val_transform=get_transform(resize=resize, scale=1, should_aug=False),
        test_transform=get_transform(resize=resize, scale=10, should_aug=False),
        batch_size=5,
        num_samples=None,
        num_workers=1,
        pin_memory=True)
    #
    # #
    # visualise(train_dl)
    # visualise(train_dl)
    visualise(train_dl)
    visualise(train_dl)
    #
    visualise(val_dl)
    visualise(val_dl)
    visualise(val_dl)
    visualise(val_dl)
    visualise(val_dl)

    visualise(test_dl)
    visualise(test_dl)
    visualise(test_dl)
    visualise(test_dl)

    # visualise(test_dl)
    # visualise(test_dl)
#
