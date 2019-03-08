import random
import torch
import glob

# import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# from fastai.vision import *
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, random_split, RandomSampler, ConcatDataset, WeightedRandomSampler
from torchvision.transforms import Resize, ToPILImage, ToTensor, Grayscale, Compose
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

random.seed(0)
import torchvision


class ImgaugWrapper():
    """
    Wrapper for imgaug
    """

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x):
        x = np.array(x)
        c, w, h = x.shape
        x = x.reshape((w, h))
        x_aug = self.aug.augment_image(x)

        return torch.from_numpy(x_aug.reshape((1, w, h)))


aug = iaa.Sometimes(0.9,
                    iaa.SomeOf((2, 3),
                               [
                                   iaa.Dropout(p=(0, 0.2)),
                                   iaa.CoarseDropout((0.05, 0.1),
                                                     size_percent=(0.01, 0.1))

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
            return dataset.imgs[idx][1]

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

    def __init__(self, scale=1.0):
        self.scale = scale

    def show_heatmap(self, x, title):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        plt.title(title)
        img_n = x.cpu().numpy().squeeze()
        sns.heatmap(img_n,
                    annot=True,
                    linewidths=.5,
                    fmt='0.2f')

    def __call__(self, x, debug=False):
        if debug: self.show_heatmap(x, 'original')

        x = x.squeeze()
        # center the patch in the middle
        x -= x[x.shape[0] // 2, x.shape[1] // 2].item()
        x *= self.scale
        x = x.unsqueeze(0)

        if debug: self.show_heatmap(x, 'center')

        return x


class TraversabilityDataset(Dataset):
    def __init__(self, df, transform, tr=None, remove_negative=False):
        self.df = pd.read_csv(df)
        if remove_negative: self.df = self.df[self.df['advancement'] >= 0]
        self.transform = transform
        self.tr = tr

        self.idx2class = {'False': 0,
                          'True': 1}

    def __getitem__(self, item):
        row = self.df.iloc[item]
        img_path = row['image_path']
        img = Image.open(img_path)

        y = row['advancement']
        y = torch.tensor(y)

        if self.tr is not None:
            y = 1 if y >= self.tr else 0

        return self.transform(img), y

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_root(cls, root, transform, tr, remove_negative):
        dfs = glob.glob(root + '/**/*-patch.csv')
        print(root + '/**/*-patch.csv')
        concat_ds = ConcatDataset([cls(df, transform, tr, remove_negative) for df in dfs])
        # needed for fastAI
        concat_ds.c = 2
        concat_ds.classes = 'False', 'True'

        return concat_ds


class FastAIImageFolder(TraversabilityDataset):
    c = 2
    classes = 'False', 'True'


def get_transform(resize, should_aug=None, scale=1):
    """
    Return a `Compose` transformation to be applied to the input of the model
    :param resize: size in pixel of the wanted final patch size
    :param should_aug: if True, dropout will be applied on the input
    :param scale: integer that is multiplied to the input
    :return:
    """
    transformations = [Grayscale()]
    if resize is not None: transformations.append(Resize((resize, resize)))
    transformations.append(ToTensor())
    if should_aug: transformations.append(ImgaugWrapper(aug))
    transformations.append(CenterAndScalePatch(scale=scale))
    return Compose(transformations)


def get_dataloaders(train_root, test_root, val_root=None, val_size=0.2, tr=0.12, num_samples=None, train_transform=None,
                    val_transform=None, test_transform=None, *args, remove_negative=False,
                    **kwargs):
    """
    Get train, val and test dataloader.
    :return: train, val and test dataloaders
    """
    print(train_transform, val_transform, test_transform)
    train_ds = FastAIImageFolder.from_root(root=train_root,
                                           transform=train_transform, tr=tr,
                                           remove_negative=remove_negative)

    train_size = int(len(train_ds) * (1 - val_size))

    if val_root is None:
        train_ds, val_ds = random_split(train_ds, [train_size, len(train_ds) - train_size])

    else:
        val_ds = FastAIImageFolder.from_root(root=val_root,
                                             transform=val_transform, tr=tr,
                                             remove_negative=remove_negative)

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
                                          transform=test_transform, tr=tr,
                                          remove_negative=remove_negative)

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

    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/home/francesco/Desktop/bars1-run-recorded/csvs-light/',
        test_root='/home/francesco/Desktop/data/92/test/',
        val_root='/home/francesco/Desktop/data/92/val',
        train_transform=get_transform(92, should_aug=True),
        val_transform=get_transform(92),
        test_transform=get_transform(92, scale=10),
        batch_size=5,
        num_samples=None,
        num_workers=1,
        pin_memory=True)

    #
    visualise(train_dl)
    visualise(train_dl)
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
