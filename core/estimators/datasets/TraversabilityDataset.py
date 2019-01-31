import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import ImageFolder

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
BATCH_SIZE = 128

transform = Compose([Grayscale(), ToTensor()])


def get_dataloaders(train_root, test_root, val_size=0.2, *args, **kwargs):
    """
    Get train and test dataloader. Due to the specific task,
    we cannot apply data-augmentation (vlip, hflip, gamma...).
    The test set is composed entirely by maps never seen by
    the model in the train set.
    :return: train, val and test dataloaders
    """
    ds = ImageFolder(root=train_root,
                     transform=transform)

    train_size = int(len(ds) * (1 - val_size))

    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])

    train_dl = DataLoader(train_ds,
                          shuffle=True,
                          *args, **kwargs)

    val_dl = DataLoader(val_ds, *args, **kwargs)

    test_ds = ImageFolder(root=train_root,
                     transform=transform)

    test_dl = DataLoader(test_ds, *args, **kwargs)

    return train_dl, val_dl, test_dl
