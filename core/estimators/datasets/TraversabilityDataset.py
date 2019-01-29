import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import ImageFolder

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
BATCH_SIZE = 128
N_WORKERS = 16

transform = Compose([Grayscale(), ToTensor()])


def get_dataloaders():
    """
    Get train and test dataloader. Due to the specific task,
    we cannot apply data-augmentation (vlip, hflip, gamma...)
    :return: train and test dataloaders
    """
    ds = ImageFolder(root='/home/francesco/Desktop/data/images-dataset' + '/images-tiny',
                     transform=transform)

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
