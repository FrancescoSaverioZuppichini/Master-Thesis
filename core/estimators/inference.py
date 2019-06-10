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

class HeightMapInference():
    """
    This class create and evaluate a given model checkpoint on a given heightmap by producing
    a texture in which each pixel x,y represent the traversable probability.
    """
    def __init__(self, hm_path, transform, learner=None, scale=1):
        self.hm_path = hm_path
        self.transform = transform
        self.learner, self.ds = learner, None
        self.scale = scale


    def __call__(self, model_dir, model_name, out_dir, texture_save_path = 'tmp', rotation=None, distance=100, *args, **kwargs):

        self.ds = InferenceDataset(self.hm_path, transform=self.transform, rotate=rotation, debug=False,*args, **kwargs)
        self.learner, _ = get_learner(model_name, model_dir, callbacks=[], dataset=self.ds, rotate=rotation)
        probs, labels = get_probs_and_labels_from_preds(self.learner.get_preds(self.learner.data.test_dl))
        path = self.ds.make_texture(probs.numpy(), labels.numpy(), 'querry-big-10',
                                    texture_save_path)

        hm_p = TraversabilityPatch.from_hm(self.ds.hm * self.scale)
        mask = read_image(path) / 255
        hm_p.plot3d_traversability(0.02, mask=mask, size=(2000, 2000),
                                     azimuth=45, elevation=45, distance=distance,
                                     save_path='{}/{}.png'.format(out_dir, rotation))

        return path

    def for_all_rotation(self, *args, **kwargs):
        for rotation in [180]:
            self(rotation=rotation, *args, **kwargs)

hm_infer = HeightMapInference('../maps/test/querry-big-10.png',
                              get_transform(None, scale=10, debug=False),
                              scale=10)

model_dir = '/media/francesco/Carino/vaevictis/data/1558825182.753924'
print(model_dir)

hm_infer.for_all_rotation(model_dir,
                          'microresnet#3-gate=7x7-n=1-se=True',
                          '/home/francesco/Documents/Master-Thesis/papers/Thesis/img/4/traversability/quarry/',
                          step=4,
                          # max_advancement = 0.66,
                          patch_size=KrockPatchExtractStrategy.patch_shape(0.70)[0],
                          texture_save_path = '/home/francesco/Documents/Master-Thesis/resources/assets/textures/0.2-0.7',

                          )
