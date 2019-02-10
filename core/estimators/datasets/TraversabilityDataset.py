import random
import torch

import numpy as np

from torch.utils.data import DataLoader, random_split, RandomSampler
from torchvision.transforms import *
from torchvision.datasets import ImageFolder

from imgaug import augmenters as iaa


class ImgaugWrapper():
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x):
        x = np.array(x)
        w, h = x.shape
        x = x.reshape((w, h))

        x_aug = self.aug.augment_image(x)

        return x_aug.reshape((w, h, 1))


aug = iaa.Sometimes(0.9,
                    iaa.SomeOf((2, 3),
                               [
                                   iaa.AdditiveGaussianNoise(scale=0.01 * 255),
                                   iaa.Dropout(p=(0, 0.2)),
                                   iaa.CoarseDropout((0.05, 0.1),
                                                     size_percent=(0.1, 0.25))

                               ], random_order=True)
                    )

import matplotlib.pyplot as plt

random.seed(0)

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
BATCH_SIZE = 128


class SampleSampler(RandomSampler):
    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(random.sample(range(n), self.num_samples))
        return iter(torch.randperm(n).tolist())


class EveryNSampler(RandomSampler):
    def __iter__(self, n_samples):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())


def get_transform(size):
    return Compose([Grayscale(), Resize((size, size)), ToTensor()])


def get_train_transform(size):
    return Compose([Grayscale(), Resize((size, size)), ImgaugWrapper(aug=aug), ToTensor()])


def get_dataloaders(train_root, test_root, val_size=0.2, num_samples=None, transform=None, train_transform=None, *args,
                    **kwargs):
    """
    Get train, val and test dataloader. Due to the specific task,
    we cannot apply data-augmentation (vlip, hflip, gamma...).
    The test set is composed entirely by maps never seen by
    the model in the train set.
    :return: train, val and test dataloaders
    """

    train_transform = transform if train_transform is None else train_transform

    print(train_transform)
    ds = ImageFolder(root=train_root,
                     transform=train_transform)

    train_size = int(len(ds) * (1 - val_size))

    # train_ds = ds
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])

    if num_samples is not None:
        train_dl = DataLoader(train_ds,
                              sampler=SampleSampler(train_ds, replacement=True, num_samples=num_samples),
                              *args, **kwargs)
    else:
        train_dl = DataLoader(train_ds,
                              shuffle=False,
                              *args, **kwargs)
    val_dl = DataLoader(val_ds, *args, **kwargs)

    test_ds = ImageFolder(root=test_root,
                          transform=transform)

    test_dl = DataLoader(test_ds, shuffle=True, *args, **kwargs)

    return train_dl, val_dl, test_dl


if __name__ == '__main__':
    train_dl, val_dl, test_dl = get_dataloaders(
        train_root='/home/francesco/Desktop/data/train/dataset/{}'.format('100-100-0.09-12-06-02-19'),
        test_root='/home/francesco/Desktop/data/test/dataset/{}'.format('100-100-0.09-12-querry'),
        val_size=0,
        train_transform=get_train_transform(100),
        transform=get_transform(100),
        batch_size=20,
        num_samples=None,
        num_workers=16,
        pin_memory=True)

    for (x, y) in test_dl:
        for i, img in enumerate(x):
            print(img.shape)
            img_n = img.cpu().numpy().squeeze()
            plt.imshow(img_n)
            # plt.show()
            plt.imsave('./{}={}.png'.format(i, str(y[i].item())), img_n)

        break
