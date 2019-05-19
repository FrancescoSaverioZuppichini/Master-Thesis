import glob
import torch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from estimators.data import *
from torch.utils.data import DataLoader, random_split, RandomSampler, ConcatDataset, WeightedRandomSampler
import torchvision
from estimators.data.transformations import RandomSimplexNoise


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
        print(label_to_count)
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
        elif dataset_type is ConcatDataset:
            return int(dataset.df['label'][idx])


    def __iter__(self):
        iter = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return iter
    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return 'ImbalancedDatasetSampler'


def visualise(dl, n=10):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    for (x, y) in dl:
        for i, img, l in zip(range(3), x, y):
            axes[i].set_title("{}".format(l))
            sns.heatmap(img.numpy().squeeze(), ax=axes[i])

        plt.show()
        break
        return


class FastAIImageFolder(TraversabilityDataset):
    c = 2
    classes = 'False', 'True'


def read_dfs(paths):
    return [pd.read_csv(path) for path in paths]


def get_dataloaders(train_root,
                    hm_root,
                    val_root=None,
                    test_root=None,
                    generate=False,
                    test_hm_root=None,
                    val_hm_root=None,
                    time_window=None,
                    val_size=0.2,
                    tr=None,
                    sampler=None,
                    num_samples=None,
                    train_transform=None,
                    train_transform_with_label=None,
                    val_transform=None,
                    test_transform=None,
                    more_than=None,
                    less_than=None,
                    down_sampling=None,
                    patch_size=None,
                    *args,
                    **kwargs):
    """
    Get train, val and test dataloader.
    :return: train, val and test dataloaders
    """

    train_meta = pd.read_csv(train_root + '/meta.csv')

    # train_meta = train_meta.drop(train_meta[(train_meta['map'] == 'slope_rocks1') & (train_meta['height'] == 1)].index)

    print('[INFO] {} simulations for training.'.format(len(train_meta)))
    if val_root is None:
        train_meta = train_meta.sample(frac=1, random_state=0)
        val_size = int((len(train_meta) // 100) * val_size)
        train_meta = train_meta[val_size:]
        val_meta = train_meta[:val_size]
        # print('[INFO] val_meta')
        # print(val_meta)
    else:
        val_meta = pd.read_csv(val_root + '/meta.csv')
    print(train_transform)
    print(train_transform_with_label)
    print(less_than)
    train_ds = FastAIImageFolder.from_meta(train_meta,
                                           train_root + '/csvs/',
                                           hm_root,
                                           patches_dir='{}/patches/{}/'.format(train_root, patch_size),
                                           tr=tr,
                                           time_window=time_window,
                                           patch_size=patch_size,
                                           more_than=more_than,
                                           less_than=less_than,
                                           transform=train_transform,
                                           transform_with_label=train_transform_with_label,
                                           down_sampling=down_sampling
                                           )

    val_root = train_root if val_root is None else val_root
    val_hm_root = hm_root if val_hm_root is None else val_hm_root

    print('[INFO] val root = {}'.format(val_root))

    val_ds = FastAIImageFolder.from_meta(val_meta,
                                         val_root + '/csvs/',
                                         val_hm_root,
                                         patches_dir='{}/patches/{}/'.format(val_root, patch_size),
                                         tr=tr,
                                         patch_size=patch_size,
                                         time_window=time_window,
                                         transform=val_transform)

    if sampler is not None:
        print('[INFO] Sampling using {} with num_samples {}'.format(sampler, num_samples))
        train_ds = TraversabilityDataset.concat_dfs(train_ds)
        train_dl = DataLoader(train_ds,
                              sampler=sampler(train_ds, num_samples=num_samples),
                              *args, **kwargs)
    else:
        train_dl = DataLoader(train_ds,
                              shuffle=True,
                              *args, **kwargs)

    val_dl = DataLoader(val_ds, shuffle=False, *args, **kwargs)

    if test_root is not None:
        print('[INFO] Using test root = {}'.format(test_root))
        test_dfs = read_dfs(glob.glob(test_root + '/csvs/' + '/*.csv'))
        test_ds = FastAIImageFolder.from_dfs(test_dfs,
                                             test_root + '/csvs/',
                                             patches_dir='{}/patches/{}/'.format(test_root, patch_size),
                                             tr=tr,
                                             time_window=time_window,
                                             transform=test_transform,
                                             patch_size=patch_size)

        test_dl = DataLoader(test_ds, shuffle=False, *args, **kwargs)
    else:
        test_dl = val_dl

    return train_dl, val_dl, test_dl


if __name__ == '__main__':
    from estimators.data.TraversabilityDataset import TraversabilityDataset
    from estimators.data.transformations import RandomSimplexNoise, DropoutAgumentation

    # simplex_noise = RandomSimplexNoise(n=50)
    import time

    start = time.time()
    meta = pd.read_csv('/media/francesco/saetta/krock-dataset/train/meta.csv')
    meta = meta[meta['map'] == 'bumps0-rocks1']
    # print('[INFO] {} simulations for training.'.format(len(meta)))
    # meta = meta[meta['map'] == 'bars1']
    print(meta)
    concat_ds = TraversabilityDataset.from_meta(meta,
                                                '/media/francesco/saetta/krock-dataset/train/csvs/',
                                                '/home/francesco/Documents/Master-Thesis/core/maps/new-train/',
                                                patches_dir='/media/francesco/saetta/krock-dataset/train/patches/0.66/',
                                                n=10,
                                                down_sampling=2,
                                                time_window=100,
                                                patch_size=0.66,
                                                tr=0.2,
                                                # transform_with_label=simplex_noise,
                                                transform=get_transform(None,
                                                                        debug=False))

    TraversabilityDataset.concat_dfs(concat_ds)
    dl = DataLoader(concat_ds, sampler=ImbalancedDatasetSampler(concat_ds), batch_size=5, pin_memory=True, num_workers=1)

    visualise(dl)
    visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
