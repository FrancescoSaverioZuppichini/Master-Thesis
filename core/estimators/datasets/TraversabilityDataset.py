import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
from torchvision.datasets import ImageFolder

from utils.postprocessing.config import Config


transform = Compose([Grayscale(), ToTensor()])

ds = ImageFolder(root=Config.IMAGES_DATASET_FOLDER + '/images', transform=transform)
print(len(ds))
img, label = ds[0]

# img.show()
print(img)