import torch
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path
from torch.nn.functional import softmax
from fastai.train import Learner, DataBunch, DatasetType
from fastai.callback import Callback
from mirror.visualisations.core import GradCam
from utils import load_model_from_name, get_learner
from patches import *
from datasets.TraversabilityDataset import get_transform, CenterAndScalePatch, TraversabilityDataset
from torch.nn.functional import softmax
from callbacks import ROC_AUC, StoreBestWorstAndSample


class GradCamVisualization():

    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_cam = GradCam(model.to(self.device), self.device)

    def __call__(self, patch):
        img = patch.unsqueeze(0)

        _, info = self.grad_cam(img, None, target_class=None)
        print(info['predictions'])

        cam = info['cam'].cpu().numpy()
        cam = cv2.resize(cam, (patch.shape[1], patch.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam *= 255

        fig = plt.figure()
        plt.title(info['predictions'])
        sns.heatmap(cam)
        plt.show()


model_dir = path.abspath('../../resources/assets/models/microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294/')

# model_dir ='/home/francesco/Desktop/carino/vaevictis/data/microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294/'
# model_dir = '/home/francesco/Documents/Master-Thesis/resources/assets/models/microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294'

model_name = 'microresnet#4-gate=3x3-n=2-se=True'

# learner = get_learner(model_name, model_dir, callbacks=[vis], root=root, transform=get_transform(None, scale=1),  tr=0.45)

model = load_model_from_name(model_dir + '/roc_auc.pth', model_name)



mod_vis = GradCamVisualization(model)

p = BarPatch((92, 92))
p(strength=0.05, offset=4, size=24)
p.plot3d()
p.hm *= 255

tr = get_transform(resize=None, debug=False)
x = tr(p.hm)

model.eval()
res = model(x.unsqueeze(0))
prob = softmax(res, dim=1)
print(prob, torch.argmax(prob, dim=1))
#
# mod_vis(x)



# #
# store_inputs = StoreBestWorstAndSample()
# #
# # root = path.abspath('../../resources/assets/datasets/test/')
# root = '/media/francesco/saetta/test/'
#
# df = root + '/df/querry-big-10/1550307709.2522066-patch.csv'
#
# ds = TraversabilityDataset(df, root=root, transform=get_transform(None, False, scale=10, debug=False), debug=True,
#                            tr=0.45)
#
# learner = get_learner(model_name, model_dir, callbacks=[store_inputs], dataset=ds)
# loss, roc = learner.validate(learner.data.test_dl, metrics=[ROC_AUC()])
#
# best  = store_inputs.df.sort_values(['output_1'], ascending=False).head(10)
# worst  = store_inputs.df.sort_values(['output_0'], ascending=False).head(30)
#
# random = vis.df_sample.head(100)
#
# print(best['output_1'], worst['output_0'])



# store_inputs.plot(worst)
#
# import cv2
# from mirror.visualisations.core import GradCam
#
# device = torch.device('cuda')
# grad_cam = GradCam(learner.model.to(device), device)
#
# def store_inputs(sample, out_dir):
#     for i, (idx, row) in enumerate(sample.iterrows()):
#         img = np.array(row['input']).squeeze()
#         img = img * 255
#         img_path = out_dir + '/{}-{}.png'.format(row['prediction'], i)
#         cv2.imwrite(img_path, img)
#
#
# def run_grad_cam(sample, out_dir):
#     for i, (idx, row) in enumerate(sample.iterrows()):
#         img = np.array(row['input'])
#         img = torch.from_numpy(img).unsqueeze(0).to(device).float()
#
#         _, info = grad_cam(img, None)
#         cam = info['cam'].cpu().numpy()
#         cam = cv2.resize(cam, (92, 92))
#         cam = (cam - cam.min()) / (cam.max() - cam.min())
#         cam *= 255
#         img_path = out_dir + '/{}-{}.png'.format(row['prediction'], i)
#         cv2.imwrite(img_path, cam)
#
#
# # store_inputs(random, '/home/francesco/Desktop/data/test-patches/patches')
# # run_grad_cam(random, '/home/francesco/Desktop/data/test-patches/textures/')
# #
# store_inputs(best, '/home/francesco/Desktop/data/test-patches/patches')
# store_inputs(worst, '/home/francesco/Desktop/data/test-patches/patches')
#
# run_grad_cam(best, '/home/francesco/Desktop/data/test-patches/textures/')
# run_grad_cam(worst, '/home/francesco/Desktop/data/test-patches/textures/')