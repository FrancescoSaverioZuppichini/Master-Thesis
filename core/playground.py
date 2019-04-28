import tqdm
import os
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from mirror.visualisations.core import GradCam
from estimators.models import *
from estimators.utils import get_learner, get_probs_and_labels_from_preds, read_patch, get_patches_form_df, \
    load_model_from_name, \
    device
from estimators.data.TraversabilityDataset import TraversabilityDataset, get_transform, PatchesDataset

from os import path
from Config import Config
from utilities.patches import *

from utilities.postprocessing.postprocessing import Handler, Compose
from utilities.pipeline import *

class StoreModelPredictionsOnDataframe(Handler):
    def __init__(self, model_name, model_dir, store_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.model_dir = model_dir
        self.dfs = []
        self.df_path2df = {}
        self.store_dir = store_dir

    def handle_dataset(self, dataset):
        df = dataset.df
        learner, _ = get_learner(self.model_name, self.model_dir, callbacks=[], dataset=dataset)
        preds = learner.get_preds(learner.data.test_dl)
        probs, labels = get_probs_and_labels_from_preds(preds)

        df['out_0'] = probs[:, 0]
        df['out_1'] = probs[:, 1]
        df['prediction'] = labels.cpu().tolist()

        return df

    def store(self):
        for df_path, df in tqdm.tqdm(self.df_path2df.items()):
            file_name = path.basename(df_path)
            map_name = path.basename(path.split(df_path)[0])
            out_path = path.normpath('{}/{}/'.format(self.store_dir, map_name))
            os.makedirs(out_path, exist_ok=True)
            out_path = path.normpath('{}/{}'.format(out_path, file_name))
            df.to_csv(out_path)

    def restore(self):
        dfs_path = glob.glob(self.store_dir + '/**/*.csv')

        for df_path in tqdm.tqdm(dfs_path):
            df = pd.read_csv(df_path)
            self.df_path2df[df_path] = df
            self.dfs.append(df)

    def __call__(self, datasets):
        bar = tqdm.tqdm(datasets)
        for dataset in bar:
            # if type(dataset) is not Data: raise ValueError(
            #     'inputs must be of type PatchesDataset')
            if hasattr(dataset, 'df_patch'): bar.set_description('[INFO] Reading {}'.format(dataset.df_path))
            if len(dataset) > 0:
                df = self.handle_dataset(dataset)
                self.dfs.append(df)
                if isinstance(dataset, PatchesDataset): df['images'] = dataset.patches
                if hasattr(dataset, 'df_patch'): self.df_path2df[dataset.df_path] = self.dfs[-1]

        return pd.concat(self.dfs)


def false_something(df, something):
    neg = df.loc[df['label'] == something]
    return neg.loc[neg['prediction'] != something]


class Best():
    name = 'best'

    def __call__(self, df):
        df = df.loc[df['prediction'] == 1]

        return df.sort_values(['out_1'], ascending=False)


class Worst():
    name = 'worst'

    def __call__(self, df):
        df = df.loc[df['prediction'] == 0]

        return df.sort_values(['out_0'], ascending=False)


class FalseNegative():
    name = 'false_negative'

    def __call__(self, df):
        return false_something(df, 0)


class FalsePositive():
    name = 'false_positive'

    def __call__(self, df):
        return false_something(df, 1)


class FilterDataframe(Handler):
    def __init__(self, filters=None, after_filter=lambda x: x, n=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_filter = after_filter
        self.filters = filters
        self.n = n

    def __call__(self, df):
        if self.filters is None: return {'all': self.after_filter(df).head(self.n)}
        res = {f.name: self.after_filter(f(df).head(self.n)) for f in self.filters}
        return res


class MergePatchesInDataframe(Handler):
    def __init__(self, image_dir, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dir = image_dir
        self.transform = transform

    def __call__(self, df):
        if 'images' not in df.columns:  df['images'] = [
            Patch.from_tensor(transform(read_patch(self.image_dir + img_path)))
            for img_path in df['image_path']]
        else: df['images'] = df['images'].apply(lambda x: Patch.from_tensor(transform(x.to_gray())))
        return df

def plot_patch(data):
    i, row = data
    patch = row['images']
    patch.plot3d('[{}] Prediction = {} Confidence = [{:2f} {:2f}] Between = {}'.format('',
                                                                                         row['prediction'],
                                                                                         row['out_0'],
                                                                                         row['out_1'],
                                                                          ""))
    return data

class PlotPatchesFromDataframe(Handler):
    def __init__(self, title='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title

    def plot_df(self, df, title=''):
        for idx, row in df.iterrows():
            print(row['out_0'],  row['out_1'])
            patch = row['images']
            patch.plot3d('[{}] Prediction = {} Confidence = [{} {}] Between = {}'.format(title,
                                                                                                 row['prediction'],
                                                                                                 row['out_0'],
                                                                                                 row['out_1'],
                                                                                                 ""))  # TODO __repr__ can be too long!
    def plot_dict(self, dict):
        for metric, df in dict.items():
            self.plot_df(df, metric)

    def __call__(self, data):
        if type(data) is dict:
            self.plot_dict(data)
        else:
            self.plot_df(data)

transform = get_transform(scale=1, debug=True)


from simulation.env.webots.krock import KrockWebotsEnv
from tf import transformations
from simulation.env.conditions import IsInside

import rospy

class RunSimulationOnPatch(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rospy.init_node("traversability_simulation")

    def __call__(self, data):
        i, row = data
        patch = row['images']

        env = KrockWebotsEnv.from_numpy(
            patch.to_gray(),
            '/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock_no_tail_patches.wbt',
            {'height': 1,
             'resolution': 0.02},
            # agent_callbacks=[RosBagSaver('~/Desktop/querry-high/bags', topics=['pose'])],
            output_path='/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock2_ros/worlds/tmp.wbt')

        tr = np.array([5, 5])

        x, y = -tr + (np.array(88) * 0.02 / 2)
        h = env.get_height(x, y)

        qto = transformations.quaternion_from_euler(0, 0, 0, axes='sxyz')

        qto = [qto[0], qto[2], qto[1], qto[3]]
        env.agent()

        init_obs = env.reset(pose=[[x, h + 0.2, y],
                                   qto], conditions=[IsInside(offset=(0.14, 0, 0))])
        # env.step(env.STOP)
        #
        # for _ in range(75):
        #     obs, r, done, _ = env.step(env.STOP)
        #     if done:
        #         break
        #
        # env.step(env.STOP)
        # env.agent.die(env)

        return data


def visualize_model_on_dataset(ds, filters, transform):
    pip = Compose([StoreModelPredictionsOnDataframe(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR),
                   FilterDataframe(filters=filters, after_filter=MergePatchesInDataframe(image_dir=Config.DATA_ROOT,
                                                                                         transform=transform)),
                   PlotPatchesFromDataframe()])

    pip([ds])


def run_model_on_patches():
    return Compose([StoreModelPredictionsOnDataframe(Config.BEST_MODEL_NAME, Config.BEST_MODEL_DIR),
                    MergePatchesInDataframe(image_dir=Config.DATA_ROOT,
                                            transform=transform),
                    ForEach([plot_patch, RunSimulationOnPatch()], make_iter=lambda x : x.iterrows())
                    ])



ds =  TraversabilityDataset(df='/media/francesco/saetta/quarry-ramp/from_flat_to_ramp/df/querry-big-10/1555314971.4928784-patch.csv',
                            root='/media/francesco/saetta/quarry-ramp/from_flat_to_ramp/',
                            tr=0.45,
                            transform=transform)
# concat = TraversabilityDataset.from_paths(Config.DATA_ROOT, [Config.DATA_DIR], tr=0.45, transform=transform)

# patches = BarPatch(shape=(88, 88), strength=0.5, size=4, offset=24)
# patches()
# patches.store('/home/francesco/Desktop/krock-test-bar/maps/test.png')
# ds = PatchesDataset([patches], transform=transform)

visualize_model_on_dataset(ds, filters=[Worst(), Best(), FalseNegative(), FalsePositive()], transform=transform)
# df = run_model_on_patches()([ds])

