import torch

from os import path
from estimators.models import zoo
from estimators.data.transformations import get_transform
from estimators.utils import get_learner
from estimators.data.InferenceDataset import InferenceDataset
from estimators.utils import get_probs_and_labels_from_preds
from torch.nn.functional import softmax
from utilities.postprocessing.utils import KrockPatchExtractStrategy
from utilities.postprocessing.handlers.functional import read_image
from utilities.patches import TraversabilityPatch
import os


class TraversabilityHeightmap(TraversabilityPatch):
    """
    This class create and evaluate a given model checkpoint on a given heightmap by producing
    a texture in which each pixel x,y represent the traversable probability.
    """

    def __init__(self, shape, model_dir, model_name, scale=1, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)
        self.hm_o = None
        self.scale = scale
        self.model_dir, self.model_name = model_dir, model_name

    def __call__(self, out_dir,
                 texture_save_path='/tmp',
                 name='',
                 step=1,
                 rotation=0, *args, **kwargs):
        os.makedirs(out_dir, exist_ok=True)

        if self.hm_o is None: self.hm_o = self.hm.copy()

        self.ds = InferenceDataset(self.hm_o, transform=get_transform(scale=self.scale), rotate=rotation, debug=False,
                                   step=step)
        self.learner, _ = get_learner(self.model_name, self.model_dir, callbacks=[], dataset=self.ds, rotate=rotation)
        probs, labels = get_probs_and_labels_from_preds(self.learner.get_preds(self.learner.data.test_dl))
        path = self.ds.make_texture(probs.numpy(), labels.numpy(), texture_save_path)

        self.hm = TraversabilityPatch.from_hm(self.hm_o * self.scale).hm
        self.mask = read_image(path) / 255
        self.save_path = '{}/{}-{}.png'.format(out_dir, name, rotation)

        return self


    def plot3d_traversability(self, pixelsize, *args, **kwargs):

        print('[INFO] saving render into {}'.format(self.save_path))
        return super().plot3d_traversability(pixelsize, mask=self.mask,
                                             save_path=self.save_path, *args, **kwargs)

hm_path = '/home/francesco/Documents/Master-Thesis/core/maps/test/sullens_cropped_resize.png'
for rotation in [0, 90, 180, 270]:

    p = TraversabilityHeightmap.from_path(hm_path,
                                          model_dir='/media/francesco/Carino/vaevictis/data/1558825182.753924',
                                          model_name='microresnet#3-gate=7x7-n=1-se=True',
                                          scale=10)(
        '/home/francesco/Documents/Master-Thesis/papers/Thesis/img/4/traversability/sullens-church/',
        texture_save_path= '/home/francesco/Documents/Master-Thesis/papers/Thesis/img/4/traversability/sullens-church/',
        step=4,
        patch_size=(78,78),
        rotation=rotation
    )
    p.plot3d_traversability(0.02, size=(1000, 1000), azimuth=45, elevation=25, distance=42)
    del p