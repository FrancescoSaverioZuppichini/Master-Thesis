import torch

from os import path
from estimators.models import zoo
from estimators.data.transformations import get_transform
from estimators.utils import get_learner
from estimators.data.InferenceDataset import InferenceDataset
from estimators.utils import get_probs_and_labels_from_preds
from torch.nn.functional import softmax

class HeightMapInference():
    """
    This class create and evaluate a given model checkpoint on a given heightmap by producing
    a texture in which each pixel x,y represent the traversable probability.
    """
    def __init__(self, hm_path, transform, learner=None):
        self.hm_path = hm_path
        self.transform = transform
        self.learner, self.ds = learner, None


    def __call__(self, model_dir, model_name, rotate=None, *args, **kwargs):

        self.ds = InferenceDataset(self.hm_path, transform=self.transform, rotate=rotate, debug=False,*args, **kwargs)
        # if self.learner is None:
        self.learner, _ = get_learner(model_name, model_dir, callbacks=[], dataset=self.ds, rotate=rotate)
        probs, labels = get_probs_and_labels_from_preds(self.learner.get_preds(self.learner.data.test_dl))
        path = self.ds.make_texture(probs.numpy(), labels.numpy(), 'querry-big-10')

        return path

    def for_all_rotation(self, *args, **kwargs):
        for rotation in [90]:
            self(rotate=rotation, *args, **kwargs)

hm_infer = HeightMapInference('../maps/test/querry-big-10.png',
                              get_transform(None, scale=20, debug=False),)

model_dir = path.abspath('/home/francesco/Desktop/carino/vaevictis/data/microresnet#4-gate=3x3-n=1-se=True-no-shift-0.001-0.66-1556484043.8083427')
print(model_dir)

hm_infer.for_all_rotation(model_dir,
                          'microresnet#4-gate=3x3-n=1-se=True',
                          step=10,
                          max_advancement = 0.66,
                        patch_size=(66, 76)
                          )
