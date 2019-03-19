import torch

from os import path
from models import zoo
from datasets.TraversabilityDataset import get_transform
from utils import get_learner

from datasets.InferenceDataset import InferenceDataset
from torch.nn.functional import softmax

class Inference():


    def __call__(self, model_dir, model_name, rotate):
        ds = InferenceDataset('../maps/test/querry-big-10.png',
                              patch_size=92,
                              step=15,
                              transform=get_transform(None, scale=10), rotate=rotate)

        model = zoo[model_name]
        self.learner = get_learner(model_name, model_dir, callbacks=[], dataset=ds)

        outs = self.learner.get_preds(self.learner.data.test_dl)

        _, preds = torch.max(outs[0], 1)
        outs = softmax(outs[0], dim=1)
        path = ds.make_texture(outs.numpy(), preds.numpy(), 'querry-big-10')
        print(path)
        return path

    def for_all_rotation(self, *args, **kwargs):
        for rotation in [0, 90, 180, 270]:
            self(rotate=rotation, *args, **kwargs)

infer = Inference()


model_dir = path.abspath('../../resources/assets/models/microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294')
print(model_dir)

infer.for_all_rotation(model_dir,
                                  'microresnet#4-gate=3x3-n=2-se=True')
# for angle in [0, 90, 180, 270]:
#     run_inference(angle)