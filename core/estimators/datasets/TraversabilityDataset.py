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
from utilities.postprocessing.postprocessing import AddAdvancement, CleanDataframe, AddHMcoordinates, \
    open_df_and_hm_from_meta_row
from utilities.postprocessing.utils import hmpatch

random.seed(0)
np.random.seed(0)

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
                                              size_percent=(0.2, 0.4))

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


class TraversabilityDataset(Dataset):
    def __init__(self, df, hm,
                 patches_dir,
                 time_window,
                 patch_size,
                 tr=None,
                 transform=None,
                 more_than=None,
                 down_sampling=None,
                 ):
        self.df = df
        self.hm = hm
        self.patches_dir = patches_dir
        self.patch_size = patch_size
        self.tr = tr
        self.transform = transform
        if down_sampling is not None: self.df = self.df[::down_sampling]
        # self.generate = generate
        self.should_generate_paths = not 'images' in df

        self.preprocess_df = Compose([AddAdvancement(time_window)])
        self.df = self.preprocess_df((df, None, None))[0]

        if more_than is not None: self.df = self.df[self.df['advancement'] >= more_than]
        if tr is not None: self.df["label"] = (self.df["advancement"] > tr)

    def generate_patch(self, row):
        patch = hmpatch(self.hm, row["hm_x"], row["hm_y"], np.rad2deg(row['pose__pose_e_orientation_z']),
                        self.patch_size,
                        scale=1)[0]

        return patch

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]

        if self.should_generate_paths:
            patch = self.generate_patch(row)
        else:
            patch = cv2.imread(self.patches_dir + '/' + row['images'])
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            patch = patch.astype(np.float32)
            patch /= 255

        y = row['advancement']

        if 'label' in self.df:
            y = row['label'].astype(np.long)

        if 'height' in row:
            patch *= row['height']

        y = torch.tensor(y)

        return self.transform(patch), y

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_meta(cls, meta, base_dir, hm_dir,  n=None, *args, **kwargs):
        datasets = []

        for (idx, row) in meta.iterrows():
            df, hm = open_df_and_hm_from_meta_row(row, base_dir, hm_dir)
            if len(df) > 0: datasets.append(cls(df, hm, *args, **kwargs))

        if n is not None: datasets = datasets[:n]
        concat_ds = ConcatDataset(datasets)

        return concat_ds

    @classmethod
    def from_root(cls, root, n=None, *args, **kwargs):
        dfs_paths = glob.glob(root + '/*.csv')
        if len(dfs_paths) == 0: dfs_paths = glob.glob(root + '/**/*.csv')

        datasets = []
        for df_path in dfs_paths:
            df = pd.read_csv(df_path)
            if len(df) > 0:
                datasets.append(cls(df, root, *args, **kwargs))

        if n is not None: datasets = datasets[:n]

        concat_ds = ConcatDataset(datasets)

        return concat_ds

    @classmethod
    def from_dfs(cls, dfs, root, *args, **kwargs):
        datasets = []

        for df in dfs:
            if len(df) > 0:
                datasets.append(cls(df, root, *args, **kwargs))

        concat_ds = ConcatDataset(datasets)
        return concat_ds


class PatchesDataset(Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
        self.df = pd.DataFrame()

    def __getitem__(self, item):
        patch = (self.patches[item].hm * 255).astype(np.uint8)
        if self.transform is not None: patch = self.transform(patch)

        return patch, 0

    def __len__(self):
        return len(self.patches)


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


def read_dfs(paths):
    return [pd.read_csv(path) for path in paths]

def get_dataloaders(train_root,
                    hm_root,
                    val_root=None,
                    test_root=None,
                    generate=False,
                    test_hm_root=None,
                    time_window=None,
                    val_size=0.2, tr=0.45,
                    num_samples=None, train_transform=None,
                    val_transform=None, test_transform=None,
                    more_than=None, should_aug=False,
                    down_sampling=None,
                    only_forward=False,
                    patch_size=None,
                    *args,
                    **kwargs):
    """
    Get train, val and test dataloader.
    :return: train, val and test dataloaders
    """


    # if generate:
    train_meta = pd.read_csv(train_root + '/bags/meta.csv')
    train_meta = train_meta[train_meta['height'] == 1]
    # train_meta = train_meta.drop(train_meta[(train_meta['map'] == 'slope_rocks1') & (train_meta['height'] == 1)].index)
    if val_root is None:
        train_meta = train_meta.sample(frac=1, random_state=0)
        val_size = int((len(train_meta) // 100) * val_size)
        # train_meta = shuffled_meta[val_size:]
        # val_meta = shuffled_meta[:val_size]
    else:
        val_meta = pd.read_csv(val_root + '/bags/meta.csv')

    train_ds = FastAIImageFolder.from_meta(train_meta,
                                           train_root + '/csvs_patches/',
                                           hm_root,
                                           patches_dir='{}/patches/{}/'.format(train_root, patch_size),
                                           time_window=time_window,
                                           transform=train_transform,
                                           tr=tr,
                                           patch_size=patch_size,
                                           more_than=more_than,
                                           down_sampling=down_sampling
                                           )

    # datasets = random.shuffle(train_ds.datasets)
    datasets = train_ds.datasets
    train_ds = ConcatDataset(datasets[val_size:])
    val_ds = ConcatDataset(datasets[:val_size])

    # val_root = train_root if val_root is None else val_root
    # val_ds = FastAIImageFolder.from_meta(val_meta,
    #                                      val_root + '/csvs_patches/',
    #                                      hm_root,
    #                                      patches_dir='{}/patches/{}/'.format(val_root, patch_size),
    #                                      time_window=time_window,
    #                                      transform=val_transform,
    #                                      tr=tr,
    #                                      patch_size=patch_size)

    # if val_root is None:
    #     if generate:
    #         val_ds = dataset_constructor(
    #                                    val_data,
    #                                    hm_root,
    #                                    time_window=time_window,
    #                                    transform=val_transform,
    #                                    tr=tr,
    #                                    patch_size=patch_size,
    #                                                )
    #     else:
    #         train_size = int(len(train_ds) - ((len(train_ds)// 100) * val_size))
    #
    #         train_ds, val_ds = random_split(train_ds, [train_size, len(train_ds) - train_size])

    # else:
    #     val_ds = dataset_constructor(root=val_root,
    #                                          transform=val_transform,
    #                                          tr=tr)

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
    val_dl = DataLoader(val_ds, shuffle=False, *args, **kwargs)
    # test_dl = val_dl
    #
    if test_root is not None:
        print('[INFO] Using test root = {}'.format(test_root))
        test_dfs = read_dfs(glob.glob(test_root + '/csvs_patches/' + '/*.csv'))
        test_ds = FastAIImageFolder.from_dfs(test_dfs,
                                            test_root + '/csvs_patches/',
                                            patches_dir='{}/patches/{}/'.format(test_root, patch_size),
                                            time_window=time_window,
                                            transform=test_transform,
                                            tr=tr,
                                            patch_size=patch_size)


        test_dl = DataLoader(test_ds, shuffle=False, *args, **kwargs)
    else:
        test_dl = val_dl

    return train_dl, val_dl, test_dl


def visualise(dl, n=10):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for (x, y) in dl:
        for i, img, l in zip(range(5), x, y):
            axes[i].set_title("{}".format(l))
            sns.heatmap(img.numpy().squeeze(), ax=axes[i])

        plt.show()
        break
        return


if __name__ == '__main__':
    import time

    start = time.time()
    meta = pd.read_csv('/media/francesco/saetta/krock-dataset/train/bags/meta.csv')
    # meta = meta[meta['height'] == 1]
    meta = meta[meta['map'] == 'bars1']
    print(meta)
    concat_ds = TraversabilityDataset.from_meta(meta,
                                                '/media/francesco/saetta/krock-dataset/train/csvs_patches/',
                                                '/home/francesco/Documents/Master-Thesis/core/maps/train/',
                                                patches_dir='/media/francesco/saetta/krock-dataset/train/patches/92',
                                                n=1,
                                                time_window=75,
                                                patch_size=92,
                                                transform=get_transform(False,
                                                                        debug=False))

    # concat_ds = TraversabilityDataset.from_('/media/francesco/saetta/krock-dataset/train/csvs_patches/',
    #                                             time_window=50 * 3,
    #                                             patch_size=None,
    #                                             tr=0.45,
    #                                             transform=get_transform(False, debug=False, scale=10))

    # hm = cv2.imread('/home/francesco/Documents/Master-Thesis/core/maps/train/slope_rocks1.png')
    # hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    #
    # concat_ds = TraversabilityDataset(pd.read_csv('/media/francesco/saetta/krock-dataset/train/csvs_parsed/slope_rocks1-1.0-0.csv'), hm,
    #                       time_window=50*3, patch_size=88, transform=get_transform(True, debug=True))
    #
    for i in range(1):
        p, y = concat_ds[i]
        print(y)
    #
    #     print(y)
    dl = DataLoader(concat_ds, batch_size=5, pin_memory=True, num_workers=1, shuffle=True)
    #
    # for batch in dl:
    #     break
    # print('Elapsed = {:.2f}'.format(time.time() - start))

    # visualise(concat_ds)
    visualise(dl)
    visualise(dl)
    visualise(dl)
    visualise(dl)
    visualise(dl)
    visualise(dl)
    visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)

#
