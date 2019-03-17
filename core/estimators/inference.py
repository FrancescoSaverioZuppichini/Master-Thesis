import torch
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rotate
from skimage.util.shape import view_as_windows

from models import zoo
from fastai.train import Learner, DataBunch
from fastai.layers import CrossEntropyFlat
from datasets.TraversabilityDataset import get_transform
from utils import get_learner

from datasets.InferenceDataset import InferenceDataset

from torch.nn.functional import softmax
class Inference():


    def __call__(self, model_dir, model_name, rotate):
        ds = InferenceDataset('/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png',
                              patch_size=92,
                              step=3,
                              transform=get_transform(None, scale=10), rotate=rotate)

        model = zoo[model_name]
        self.learner = get_learner(model_name, model_dir, callbacks=[], dataset=ds)

        outs = self.learner.get_preds(self.learner.data.test_dl)

        _, preds = torch.max(outs[0], 1)
        outs = softmax(outs[0], dim=1)
        path = ds.make_texture(outs.numpy(), preds.numpy(), 'querry')

        return path

    def for_all_rotation(self, *args, **kwargs):
        for rotation in [0, 90, 180, 270]:
            self(rotate=rotation, *args, **kwargs)

infer = Inference()

infer.for_all_rotation('microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294',
                                  'microresnet#4-gate=3x3-n=2-se=True')
# for angle in [0, 90, 180, 270]:
#     run_inference(angle)