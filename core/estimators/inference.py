import torch

from os import path
from models import zoo
from datasets.TraversabilityDataset import get_transform
from utils import get_learner
from datasets.InferenceDataset import InferenceDataset
from utils import get_probs_and_labels_from_preds
from torch.nn.functional import softmax

class HeightMapInference():
    """
    This class create and evaluate a given model checkpoint on a given heightmaps by producing
    a texture in which each pixel x,y represent the traversable probability.
    """
    def __init__(self, hm_path, transform):
        self.hm_path = hm_path
        self.transform = transform
        self.learner, self.ds = None, None


    def __call__(self, model_dir, model_name, rotate=None, *args, **kwargs):
        if self.learner is not None:
            self.ds.rotate = rotate
        else:
            self.ds = InferenceDataset(self.hm_path, transform=self.transform, rotate=rotate, *args, **kwargs)
            self.learner = get_learner(model_name, model_dir, callbacks=[], dataset=self.ds)

        probs, labels = get_probs_and_labels_from_preds(self.learner.get_preds(self.learner.data.test_dl))
        path = self.ds.make_texture(probs.numpy(), labels.numpy(), 'querry-big-10')
        # x = torch.ones((1, 1, 92, 92))
        # res = self.learner.model(x).float()

        # print(softmax(res, dim=1))

        return path

    def for_all_rotation(self, *args, **kwargs):
        for rotation in [0, 90, 180, 270]:
            self(rotate=rotation, *args, **kwargs)

hm_infer = HeightMapInference('../maps/test/querry-big-10.png', get_transform(None, scale=10))

model_dir = path.abspath('../../resources/assets/models/microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294')
print(model_dir)

hm_infer.for_all_rotation(model_dir,
                       'microresnet#4-gate=3x3-n=2-se=True',
                       step=10,
                       patch_size=92)
