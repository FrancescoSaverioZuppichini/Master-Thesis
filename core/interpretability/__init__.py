import rospy
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import cv2
import seaborn as sns

from utilities.patches import *
from utilities.postprocessing.utils import KrockPatchExtractStrategy

from estimators.data.TraversabilityDataset import TraversabilityDataset
from estimators.data.transformations import get_transform
from estimators.utils import get_learner, get_probs_and_labels_from_preds
from utilities.postprocessing.postprocessing import Bags2Dataframe, ParseDataframe, AddAdvancement
from utilities.pipeline import *
from torch.utils.data import ConcatDataset
from simulation.env.webots.krock import KrockWebotsEnv
from simulation.env.spawn import spawn_points2webots_pose
from simulation.agent.callbacks import RosBagSaver
from mirror.visualisations.core import GradCam
from functools import partialmethod, partial
from .filters import *
from fastai.vision import ClassificationInterpretation

# Utils
def dataset2dataframe(ds):
    df = pd.DataFrame(data={'out_0': [], 'out_1': [], 'prediction': []})
    if type(ds) is ConcatDataset:
        for ds in ds.datasets:
            if df is None:
                df = ds.df
            else:
                df = pd.concat([df, ds.df])
        df = df.reset_index(drop=True)
    elif type(ds) is TraversabilityDataset:
        df = ds.df

    return df


def outs2dafaframe(df, probs, labels):
    df['out_0'] = probs[:, 0]
    df['out_1'] = probs[:, 1]
    df['prediction'] = labels.cpu().tolist()
    return df

class ExplainModel():
    def __init__(self, ds, df, learner):
        self.ds = ds
        self.df = df
        self.learner = learner

    def by_looking_at(self, filters=None, how_many=10):
        if filters is None: return self
        return {f.name: ExplainModel(*self.zip_df_ds(f(self.df).head(how_many)), self.learner) for f in filters}

    def tell(self, answer):
        return answer(self)

    def zip_df_ds(self, df):
        indx = df.index.values
        ds = [self.ds[i] for i in indx]
        return ds, df

    @classmethod
    def from_preds(cls, preds, ds, learner, *args, **kwargs):
        probs, labels = get_probs_and_labels_from_preds(preds)
        df = dataset2dataframe(ds)
        df = outs2dafaframe(df, probs, labels)
        print(len(df))
        return cls(ds, df, learner)

    @classmethod
    def from_leaner(cls, learner, ds, *args, **kwarg):
        preds = learner.get_preds()
        return cls.from_preds(preds, ds, learner, *args, **kwarg)


class KrockWebotsPatches(KrockWebotsEnv):
    world_path = '/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock_no_tail_patches.wbt'
    output_dir = '/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock2_ros/worlds'
    bags_out_dir = '/tmp/'
    bag_name = 'patches_bag'

    def reset(self):
        y = self.y_dim // 2
        x = 36
        print(x, y)
        pose = spawn_points2webots_pose([x, y], self, orientation=[0, 0, 0, 0])
        super().reset(pose=pose)

    def run(self, for_seconds):
        self.reset()
        elapsed = 0
        start = time.time()
        while elapsed <= for_seconds:
            elapsed = time.time() - start
            obs, r, done, _ = self.step(self.GO_FORWARD)
        self.agent.die(self, self.bag_name)
        obs, r, done, _ = self.step(self.STOP)

        return self.bag_name

    @classmethod
    def from_patch(cls, patch, *args, **kwargs):
        rospy.init_node("patches_simulation")
        return cls.from_numpy(patch.hm * 255,
                              cls.world_path,
                              {'height': 1,
                               'resolution': 0.02},
                              agent_callbacks=[RosBagSaver(cls.bags_out_dir, topics=['pose'])],
                              output_path=cls.output_dir + '/tmp.wbt')


def run_patch_on_simulator(p, for_seconds):
    env = KrockWebotsPatches.from_patch(p)
    return env.run(for_seconds)


class WebotsRunnablePatch(Patch):
    def run_on_simulator(self, for_seconds):
        self.env = KrockWebotsPatches.from_patch(self)
        self.bag_name = self.env.run(for_seconds)
        return self

    def get_advancement(self, time_window):
        return Compose([
            Bags2Dataframe(self.env.bags_out_dir + '/'),
            ParseDataframe(),
            AddAdvancement(time_window),
            lambda x: x[0],
            lambda df: df.reset_index(drop=True),
            lambda df: df['advancement'][0]
        ])(self.bag_name)


class GrandCamAnswarable():
    def grad_cam(self, module, device, *args, **kwargs):
        x = torch.tensor(self.hm)
        x = x.unsqueeze(0).unsqueeze(0).to(device)
        x = GradCam(module, device)(x, *args, **kwargs)
        x = x[1]['cam']
        return Patch.from_hm(x / 255)


class PatchAnswer(WebotsRunnablePatch, GrandCamAnswarable, HeatMapShowable):
    def __init__(self, patch_size, info, *args, **kwargs):
        super().__init__(patch_size, *args, **kwargs)
        self.info = info

    @property
    def title(self):
        pred = self.info['prediction']
        title = 'prediction = {} - {:.2f} - {:.2f}'.format(pred, self.info['out_0'], self.info['out_1'])
        if 'label' in self.info:
            title += '\nground truth = {}'.format(int(self.info['label']))
        if 'advancement' in self.info:
            title += '\nadvancenment = {:.2f}'.format(self.info['advancement'])
        return title

    def add_advancement_using_simulator(self, for_seconds, time_window):
        self.run_on_simulator(for_seconds)
        adv = self.get_advancement(time_window)
        self.info['advancement'] = adv
        return self

    @classmethod
    def from_explain(cls, explain):
        return [cls.from_tensor(el[0], row) for el, (dx, row) in zip(explain.ds, explain.df.iterrows())]



class ClassificationAnswer(ClassificationInterpretation):

    @classmethod
    def from_explain(cls, explain):
        return cls.from_learner(explain.learner)