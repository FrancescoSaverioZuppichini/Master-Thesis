import random
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import ImageFolder

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

def get_transform(size):
    return Compose([Grayscale(), Resize((size, size)), ToTensor()])


def get_dataloaders(train_root, test_root, val_size=0.2, num_samples=None, transform=None, *args, **kwargs):
    """
    Get train, val and test dataloader. Due to the specific task,
    we cannot apply data-augmentation (vlip, hflip, gamma...).
    The test set is composed entirely by maps never seen by
    the model in the train set.
    :return: train, val and test dataloaders
    """
    ds = ImageFolder(root=train_root,
                     transform=transform)

    train_size = int(len(ds) * (1 - val_size))

    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])

    if num_samples is not None:
        train_dl = DataLoader(train_ds,
                              sampler=SampleSampler(train_ds, replacement=True, num_samples=num_samples),
                              *args, **kwargs)
    else:
        train_dl = DataLoader(train_ds,
                              shuffle=True,
                              *args, **kwargs)
    val_dl = DataLoader(val_ds, *args, **kwargs)

    test_ds = ImageFolder(root=test_root,
                     transform=transform)

    test_dl = DataLoader(test_ds, *args, **kwargs)

    return train_dl, val_dl, test_dl
