import torch
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows

from models.omar_cnn import *
from fastai.train import Learner, DataBunch
from fastai.layers import CrossEntropyFlat
from datasets.TraversabilityDataset import get_transform
from models.custom_resnet import *


class InferenceDataset(Dataset):
    def __init__(self, hm_path, patch_size=80, step=1, transform=None):
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
        # self.hm = self.hm.astype(np.float32)
        self.images = view_as_windows(self.hm, (patch_size, patch_size), step)
        self.images_shape = self.images.shape

        self.images = self.images.reshape(-1, patch_size, patch_size)
        self.transform = transform
        self.step = step
        self.patch_size = patch_size

    def __getitem__(self, item):
        img = Image.fromarray(self.images[item])

        # plt.imshow(self.images[item])
        # plt.show()
        if self.transform: img = self.transform(img)



        return img, 0

    def __len__(self):
        return len(self.images)

    def visualise(self, predictions):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        sns.heatmap(self.hm, ax=ax)

        predictions = predictions.reshape(self.images_shape[0], self.images_shape[1])

        print(predictions.shape)
        w, h = self.hm.shape

        buffer = []

        j = 0
        for x in range(0, w, self.step):
            i = 0
            for y in range(self.step, h, self.step):
                try:
                    pred = predictions[i,j]
                    color = 'r' if pred == 0 else 'g'
                    buffer.append((x, y-self.step, color))

                    if color == 'g':
                        rect = mpatches.Rectangle((x, y), self.patch_size,
                                                  self.patch_size, linewidth=0, edgecolor='none', facecolor='b', alpha=0.01)
                        ax.add_patch(rect)
                    # ax.plot(x + self.patch_size // 2 , y + self.patch_size //2, marker='o', color=color,  alpha=0.2)
                    i+=1
                except IndexError:
                    break
            j += 1

        red_buffer = filter(lambda x: x[2] == 'r', buffer)

        green_buffer = filter(lambda x :x[2] == 'g', buffer)

        def draw_rects(buffer):
            for x, y, color in buffer:
                rect = mpatches.Rectangle((x, y), self.patch_size,
                                          self.patch_size, linewidth=0, edgecolor='none', facecolor=color)
                ax.add_patch(rect)

        # draw_rects(green_buffer)
        # draw_rects(red_buffer)

        plt.show()

ds = InferenceDataset('/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png',
                      step=1,
                      transform=get_transform(64, scale=1))

dl = DataLoader(ds, batch_size=128, num_workers=16, shuffle=False)


data = DataBunch(train_dl=dl, valid_dl=dl, test_dl=dl)

model = OmarCNN()
# model = MicroResnet.micro(1,
#                           n_classes=2,
#                           block=[BasicBlock, BasicBlock, BasicBlock, BasicBlockSE],
#                           preactivated=True)
criterion = CrossEntropyFlat()

learner = Learner(data=data,
                  model=model)

learner.load('/home/francesco/Desktop/carino/vaevictis/data/omar-100-100-0.12-12-no_tail-spawn-0.001-64-accuracy')

outs = learner.get_preds(data.test_dl)

_, preds = torch.max(outs[0], 1)

ds.visualise(preds.numpy())

