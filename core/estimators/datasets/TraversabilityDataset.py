import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from torchvision.datasets import ImageFolder

from utils.postprocessing.config import Config

import matplotlib.pyplot as plt

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
BATCH_SIZE = 64
N_WORKERS = 16
transform = Compose([Grayscale(), Resize((80, 80)), ToTensor()])

def get_dataloaders():

    ds = ImageFolder(root=Config.IMAGES_DATASET_FOLDER + '/images', transform=transform)

    train_size = int(len(ds) * TRAIN_SIZE)

    train_ds, test_ds = random_split(ds, [train_size, len(ds) - train_size])

    train_dl = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          num_workers=N_WORKERS,
                          shuffle=True)

    test_dl = DataLoader(test_ds,
                          batch_size=BATCH_SIZE,
                          num_workers=N_WORKERS)

    return train_dl, test_dl


if __name__ == '__main__':
    # ds = ImageFolder(root=Config.IMAGES_DATASET_FOLDER + '/images', transform=transform)
    # train_size = int(len(ds) * TRAIN_SIZE)
    #
    # train_ds, test_ds = random_split(ds, [train_size, len(ds) - train_size])
    # for i in range(10):
    #     print(train_ds[i][1])
    dl, _ = get_dataloaders()
    for batch in dl:
        imgs, targets = batch
        for img, target in zip(imgs, targets):
            plt.imshow(img.cpu().numpy().squeeze())
            plt.title(str(target))
            plt.show()