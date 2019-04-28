import glob

import pandas as pd

from estimators.data import *
from estimators.utils import visualise
from torch.utils.data import DataLoader, random_split, RandomSampler, ConcatDataset, WeightedRandomSampler

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


    # train_meta = train_meta[train_meta['height'] == 1]
    train_meta = train_meta.drop(train_meta[(train_meta['map'] == 'slope_rocks1') & (train_meta['height'] == 1)].index)

    print('[INFO] {} simulations for training.'.format(len(train_meta)))
    print(train_meta)
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

    if num_samples is not None:
        print('[INFO] Sampling')
        train_dl = DataLoader(train_ds,
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


if __name__ == '__main__':
    import time

    start = time.time()
    meta = pd.read_csv('/media/francesco/saetta/krock-dataset/train/bags/meta.csv')
    meta = meta[meta['map'] == 'bars1']
    print('[INFO] {} simulations for training.'.format(len(meta)))
    # meta = meta[meta['map'] == 'bars1']
    print(meta)
    concat_ds = TraversabilityDataset.from_meta(meta,
                                                '/media/francesco/saetta/krock-dataset/train/csvs_patches/',
                                                '/home/francesco/Documents/Master-Thesis/core/maps/train/',
                                                patches_dir='/media/francesco/saetta/krock-dataset/train/patches/92',
                                                n=1,
                                                time_window=75,
                                                patch_size=92,
                                                transform=get_transform(True,
                                                                        debug=True))

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
    dl = DataLoader(concat_ds, batch_size=5, pin_memory=True, num_workers=1, shuffle=False)
    #
    # for batch in dl:
    #     break
    # print('Elapsed = {:.2f}'.format(time.time() - start))

    # visualise(concat_ds)
    visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
    # visualise(dl)
