import torch

from os import path
from estimators.models import zoo
from estimators.data.transformations import get_transform
from estimators.utils import get_learner
from estimators.data.InferenceDataset import InferenceDataset
from estimators.utils import get_probs_and_labels_from_preds
from torch.nn.functional import softmax
from utilities.postprocessing.utils import KrockPatchExtractStrategy
from utilities.patches import Patch

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


    def __call__(self, model_dir, model_name, rotation=None, *args, **kwargs):

        self.ds = InferenceDataset(self.hm_path, transform=self.transform, rotate=rotation, debug=False,*args, **kwargs)
        # if self.learner is None:
        self.learner, _ = get_learner(model_name, model_dir, callbacks=[], dataset=self.ds, rotate=rotation)
        probs, labels = get_probs_and_labels_from_preds(self.learner.get_preds(self.learner.data.test_dl))
        path = self.ds.make_texture(probs.numpy(), labels.numpy(), 'querry-big-10',
                                    '/home/francesco/Documents/Master-Thesis/resources/assets/textures/0.2-0.66/')

        hm_p = Patch.from_hm(self.ds.hm * self.scale)
        # hm_p.plot3d_mayavi(0.02, size=(1000,1000), azimuth=45, elevation=45, distance=85, save_path='/home/francesco/Desktop/{}.png'.format(rotation),
        #              texture_path=path, mesh=True, color=(1,1,1))
        return path

    def for_all_rotation(self, *args, **kwargs):
        for rotation in [270]:
            self(rotation=rotation, *args, **kwargs)

hm_infer = HeightMapInference('../maps/test/querry-big-10.png',
                              get_transform(None, scale=10, debug=False),
                              scale=10)

model_dir = '/media/francesco/Carino/vaevictis/data/1558825182.753924'
print(model_dir)

hm_infer.for_all_rotation(model_dir,
                          'microresnet#3-gate=7x7-n=1-se=True',
                          step=10,
                          # max_advancement = 0.66,
                          patch_size=KrockPatchExtractStrategy.patch_shape(0.70)[0]
                          )
