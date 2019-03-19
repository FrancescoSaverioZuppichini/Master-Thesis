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
from datasets.TraversabilityDataset import get_transform, CenterAndScalePatch
from torch.nn.functional import softmax
class Visualise(Callback):
    """
    Store each input, target and prediction into a Dataframe in order to
    perform custom queries on the dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = None
        self.df_sample = None

    def on_batch_end(self, last_input, last_output, last_target, train, **kwargs):
        if not train:
            to_np = lambda x: x.cpu().numpy()
            last_target = to_np(last_target)
            last_output_act = softmax(last_output, dim=1)
            prediction = to_np(torch.argmax(last_output_act, dim=1))
            last_output_act = to_np(last_output_act)

            output_0, output_1 = last_output_act[:,0].tolist(), last_output_act[:,1].tolist()

            df = pd.DataFrame(data={'input': to_np(last_input).tolist(),
                                    'output_0': output_0,
                                    'output_1': output_1,
                                    'prediction': prediction.tolist(),
                                    'target': last_target.tolist() })

            if self.df is None:
                self.df = df
                self.df_sample = df.sample(1)
            else:
                self.df = pd.concat([self.df, df])
                self.df_sample = pd.concat([self.df_sample, df.sample(1)])

            self.free_memory()


    def free_memory(self):
        """
        This function reduces the number of rows in the dataframe.
        If we store everything we will run out of RAM!
        :return:
        """

        best = self.df.sort_values(['output_1'], ascending=False).head(10)
        worst = self.df.sort_values(['output_0'], ascending=False).head(10)

        self.df = pd.concat([best, worst])

    def plot(self, sample):
        for img, pred in zip(sample['input'], sample['prediction']):
            img = np.array(img).squeeze()
            plt.title(pred)
            sns.heatmap(img,
                        vmin=0,
                        # annot=True,
                        # linewidths=.5,
                        fmt='0.2f')
            plt.show()

class GradCamVisualization():

    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_cam = GradCam(model.to(self.device), self.device)

    def __call__(self, patch):
        img = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device).float()

        _, info = self.grad_cam(img, None, target_class=None)
        print(info['predictions'])

        cam = info['cam'].cpu().numpy()
        cam = cv2.resize(cam, patch.shape)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam *= 255

        fig = plt.figure()
        plt.title(info['predictions'])
        sns.heatmap(cam)
        plt.show()




model_dir = 'microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552506972.5252423'
model_name = 'microresnet#4-gate=3x3-n=2-se=True'
model_path = path.abspath('../../resources/assets/models/microresnet#4-gate=3x3-n=2-se=True-750-0.001-None-1552582563.7411294')

# learner = get_learner(model_name, model_dir, callbacks=[vis], root=root, transform=get_transform(None, scale=1),  tr=0.45)

model = load_model_from_name(model_path + '/roc_auc.pth', model_name)


mod_vis = GradCamVisualization(model)

p = HolesPatch((92,92))
p(strength=0)
p.plot2d()

tr = CenterAndScalePatch(debug=True)

res = model(torch.zeros((1,1,92,92)).float())
print(softmax(res, dim=1))
# mod_vis(tr(p.hm))
#
# vis = Visualise()
# # root = '/home/francesco/Desktop/data/750/test/df/'
# root = '/home/francesco/Desktop/data/750/train/df/bars1/'
# learner = get_learner(model_name, model_dir, callbacks=[vis], root=root, transform=get_transform(None, scale=1),  tr=0.45)
#
#
# # loss, acc, roc = learner.validate(learner.data.test_dl, metrics=[accuracy, ROC_AUC()])
# import pandas as pd
#
# preds, targs = learner.get_preds(ds_type=DatasetType.Valid)
#
# preds = softmax(preds, dim=1)
# preds = torch.argmax(preds, dim=1)
# # def show_preds(leaner):
#
# best  = vis.df.sort_values(['output_1'], ascending=False).head(30)
# worst  = vis.df.sort_values(['output_0'], ascending=False).head(30)
#
# random = vis.df_sample.head(100)
#
# print(best['output_1'], worst['output_0'])
#
# # vis.plot(best)
# #
# # vis.plot(worst)
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