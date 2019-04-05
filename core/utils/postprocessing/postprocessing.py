# from postprocessing import *
import os
import os
import glob
import cv2
import dateutil
import rosbag_pandas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import path
from utils import *
from pypeln import thread as th


class PostProcessingConfig():
    def __init__(self, maps_folder, patch_size, advancement_th, time_window, skip_every, translation,
                 resolution=0.02, scale=1, n_workers=16,
                 base_dir=None, csv_dir=None, out_dir=None, patch_dir=None, verbose=True, patches=True, name=''):
        self.maps_folder, self.base_dir, self.out_dir, self.csv_dir = maps_folder, base_dir, out_dir, csv_dir

        self.bags_dir = self.base_dir + '/bags/'

        self.patch_size, self.advancement_th, self.time_window = patch_size, advancement_th, time_window
        self.scale, self.skip_every = scale, skip_every
        self.resolution, self.translation = resolution, translation
        self.n_workers = n_workers
        self.name = name
        if out_dir is None: self.out_dir = base_dir

        self.out_dir = os.path.normpath(self.out_dir + name)

    @property
    def dataset_name(self):
        return '/{}'.format(self.patch_size)

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))


class Handler():
    def __init__(self, successor=None):
        self.successor = successor

    def __call__(self, *args, **kwargs):
        res = self.handle(*args, **kwargs)

        if self.successor is not None: res = self.successor(res)

        return res

    def handle(self, *args, **kwargs):
        raise NotImplementedError

    def restore(self, *args, **kwargs):
        raise NotImplementedError


class MultiThreadWrapper():
    def __init__(self, n_workers):
        self.n_workers = n_workers

    def __call__(self, pip, data):
        return list(tqdm(th.map(pip, data, workers=self.n_workers), total=len(data)))


class PostProcessingHandler(Handler):
    def __init__(self, config: PostProcessingConfig, successor=None):
        super().__init__(successor=successor)
        self.config = config


def make_path(file_path, out_dir):
    splitted = file_path.split('/')
    map_name, file_name = splitted[-2], splitted[-1]
    return path.normpath(
        '{}/{}/{}'.format(out_dir + '/csvs/', map_name, path.splitext(file_name)[0]))


class BagsHandler(PostProcessingHandler):
    """
    This class loads the bags file and converted them to Pandas' dataframe. In addition,
    it opens each map for each file and add it to the return tuple.
    """

    def handle(self, file_name):
        df = rosbag_pandas.bag_to_dataframe(file_name)

        map_name = filename2map(file_name)

        return (df, map_name, file_name)


class InMemoryHandler(PostProcessingHandler):
    """
    This class loads only the maps without using the bags file. This must be used when
    the csvs files from DataFrameHandler where already generated
    """

    def handle(self, file_name):
        map_name = filename2map(file_name)
        df = pd.read_csv(file_name)

        map_path = '{}/{}.png'.format(self.config.maps_folder, map_name)

        hm = read_image(map_path)
        return (df, hm, file_name)


class DataFrameHandler(PostProcessingHandler):
    """
    This class decorate the dataframe generated from the bags file with
    all the information we need, e.g. 'advancement'
    """

    def df_convert_date2timestamp(self, df):
        """
        Convert the index column of the given dataframe which contains the converted bag time
        to a time expressed in second starting from the beginning. For example,
        1549572176.951121,
        1549572176.8528721

        becomes

        0.0,
        0.1

        :param df:
        :return:
        """
        # df = df.reset_index()  # we need to drop the index since the time column is used for it
        df['timestamp'] = df.index.apply(lambda x: dateutil.parser.parse(x).timestamp())
        df = df.set_index(df['timestamp'])  # reset back the index to the time

        return df

    def df_add_hm_coords(self, df, hm):
        """
        Decorate the current dataframe with the robot's position
        in the heightmap. The x and y coordinates will be stored in 'hm_x' and 'hm_y' columns.
        :param df: current dataframe
        :param hm: heightmap representing the map
        :return: decorated dataframe
        """

        df[['hm_x', 'hm_y']] = df.apply(
            lambda x: to_hm_coordinates(x, hm, self.config.resolution, self.config.translation),
            axis=1)
        return df

    def extract_cos_sin(self, df):
        df["S_oX"] = np.cos(df['pose__pose_e_orientation_z'].values)
        df["S_oY"] = np.sin(df['pose__pose_e_orientation_z'].values)

        assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))

        return df

    def df_clean_by_dropping(self, df, max_x, max_y):
        """
        Clean the given dataframe by dropping the rows
        - with time stamp < 1 and > 19 seconds
        - where the robot is upside down
        :param df:
        :param max_x:
        :param max_y:
        :return:
        """
        df = df.loc[df.index >= 1].dropna()
        # df = df.loc[df.index <= 19].dropna()
        # robot upside down
        df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()
        return df



    def handle(self, data):
        df, map_name, file_path = data

        def make_path(file_path):
            splitted = file_path.split('/')
            map_name, file_name = splitted[-2], splitted[-1]
            return path.normpath(
                '{}/{}/{}'.format(self.config.base_dir + '/csvs/', map_name, path.splitext(file_name)[0]))

        map_name = filename2map(file_path)
        map_path = '{}/{}.png'.format(self.config.maps_folder, map_name)

        hm = read_image(map_path)
        file_path = make_path(file_path)

        try:
            if path.isfile(file_path + '-complete.csv'):
                print('file exist, loading...')
                df = pd.read_csv(file_path + '-complete.csv')
            #
            else:
                df = df_convert_date2timestamp(df)
                df = df_convert_quaterion2euler(df)
                df = self.df_clean_by_dropping(df, hm.shape[0] * self.config.resolution,
                                               hm.shape[1] * self.config.resolution)


                if len(df) > 0:
                    df = self.extract_cos_sin(df)
                    df = self.df_add_hm_coords(df, hm)
                    # df = self.df_clean_by_removing_outliers(df, hm)
                    df = df.dropna()
                    # TODO add flag to decide if store the csv or not
                    os.makedirs(path.dirname(file_path), exist_ok=True)
                    df.to_csv(file_path + '-complete.csv')
                # else:
                # print('{} contains 0 rows, dropping...'.format(file_path))

        except Exception as e:
            print('DATAFRAME')
            print(e)
            pass

        return df, hm, file_path


class PatchesHandler(PostProcessingHandler):

    def __init__(self, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug

    def df_clean_by_removing_outliers(self, df, hm):
        offset = 22

        index = df[(df['hm_y'] > (hm.shape[0] - offset)) | (df['hm_y'] < offset)
                   | (df['hm_x'] > (hm.shape[1] - offset)) | (df['hm_x'] < offset)
                   ].index

        # print('removing {} outliers'.format(len(index)))
        # if there are some outliers, we remove all the rows after the first one
        if len(index) > 0:
            idx = index[0]
            df = df.loc[0:idx]

        return df

    def df_add_label(self, df, advancement_th):
        """
        Decore the dataframe with the 'label' column that indicates
        if a patch is traversable or not based on the advancement
        :param df:
        :param advancement_th:
        :return:
        """
        df["label"] = df["advancement"] > advancement_th
        return df

    def df_add_advancement(self, df, dt):
        """
        Project the distance x and y computed using a rolling window
        into the current line to compute the advancement
        :param df:
        :param dt:
        :return:
        """
        # get out the cos and sin component from the euler's w angle

        # look dt in the future and compute the distance for booth axis
        df["S_dX"] = df.rolling(window=(dt + 1))['pose__pose_position_x'].apply(lambda x: x[-1] - x[0], raw=True).shift(
            -dt)
        df["S_dY"] = df.rolling(window=(dt + 1))['pose__pose_position_y'].apply(lambda x: x[-1] - x[0], raw=True).shift(
            -dt)
        # compute euclidean distance
        # df["S_d"] = np.linalg.norm(df[["S_dX", "S_dY"]], axis=1)
        # project x and y in the current line and compute the advancement
        df["advancement"] = np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

        return df


    def df_adjust_robot_center(self, df):
        # df['hm_x'] = df['hm_x'] - 14

        return df

    def handle(self, data):
        """
        Given a dataframe, and heightmap and the file path, this function extracts a patch
        every `Config.SKIP_EVERY` rows. This is done due to the big rate that we used to
        store the data, around 250hz. A `Config.SKIP_EVERY=12` is equal to a rate of 20.
        :param data:

        :return:
        """
        df, hm, file_path = data
        if len(df) <= 0:  return data
        dirs, name = path.split(file_path)

        out_dir = self.config.out_dir
        _, map_name = path.split(dirs)

        file_path_light = path.normpath(self.config.out_dir + '/df/' + map_name)

        os.makedirs(out_dir + '/patches', exist_ok=True)
        os.makedirs(file_path_light, exist_ok=True)
        # df = self.df_add_label(df, self.config.advancement_th)
        df = self.df_clean_by_removing_outliers(df, hm)
        df = self.df_add_advancement(df, self.config.time_window)
        df = df.dropna()

        df = df[::self.config.skip_every]

        df = self.df_adjust_robot_center(df)

        image_paths = []

        to_show = 1
        for idx, (i, row) in enumerate(df.iterrows()):
            patch = \
                hmpatch(hm, row["hm_x"], row["hm_y"], np.rad2deg(row['pose__pose_e_orientation_z']),
                        self.config.patch_size,
                        scale=1)[0]

            if self.debug and to_show > idx:
                fig = plt.figure()
                sns.heatmap(patch, vmin=0, vmax=1)
                plt.title('before saving')
                plt.show()

            patch = (patch * 255).astype(np.uint8)

            image_path = path.normpath('{}/patches/{}.png'.format(out_dir, row['timestamp']))

            cv2.imwrite(image_path, patch)

            image_paths.append('/patches/{}.png'.format(row['timestamp']))  # store only name not abs path

            if self.debug and to_show > idx:
                fig = plt.figure()
                patch_saved = cv2.imread(image_path)
                patch_saved = cv2.cvtColor(patch_saved, cv2.COLOR_BGR2GRAY)

                sns.heatmap(patch_saved / 255, vmin=0, vmax=1)
                plt.title('after saving')
                plt.show()

        df['image_path'] = image_paths
        df['map_name'] = map_name
        # TODO the name is wrong!
        # create a new small dataframe with the reference to the image stored
        df.to_csv(file_path_light + '/{}-patch.csv'.format(name.split('-')[0]))

        return data



def make_and_run_chain(config, memory=True):
    patches_h = PatchesHandler(config=config, debug=False)
    df_h = DataFrameHandler(successor=patches_h, config=config)
    b_h = BagsHandler(config=config, successor=df_h)

    files = glob.glob('{}/csvs/**/*.csv'.format(config.base_dir))

    if len(files) > 0:
        print('[INFO] Loading from memory.')
        pip = InMemoryHandler(config=config, successor=patches_h)
    else:
        print('[INFO] Loading bags.')
        files = glob.glob('{}/**/*.bag'.format(config.bags_dir))
        pip = b_h

    th_wrap = MultiThreadWrapper(14)
    data = th_wrap(pip, files)

    return data


def run_train_val_test_chain(base_dir, base_maps_dir, *args, **kwargs):
    for name in ['train', 'val', 'test']:
        maps_dir = base_maps_dir + '/{}'.format(name)
        config = PostProcessingConfig(base_dir=base_dir + '/{}'.format(name), maps_folder=maps_dir, name=name, *args,
                                      **kwargs)

        make_and_run_chain(config)


if __name__ == '__main__':
    run_train_val_test_chain(base_dir='/media/francesco/saetta/krock-dataset/92/',
                             base_maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/',
                             out_dir='/media/francesco/saetta/no-shift-88-750/',
                             patch_size=88,
                             advancement_th=0.45,
                             skip_every=12,
                             translation=[5, 5],
                             time_window=750)
    # #
    # config = PostProcessingConfig(base_dir='/home/francesco/Desktop/krock-center-tail/',
    #                         maps_folder='/home/francesco/Desktop/krock-center-tail/',
    #                          patch_size=88,
    #                          advancement_th=0.45,
    #                          skip_every=12,
    #                          translation=[5, 5],
    #                          time_window=100)
    # make_and_run_chain(config)
    #

    # config = PostProcessingConfig(base_dir='/media/francesco/saetta/krock-dataset/92/val_new/',
    #                               maps_folder='/home/francesco/Documents/Master-Thesis/core/maps/val/',
    #                               patch_size=88,
    #                               out_dir='/media/francesco/saetta/no-shift-88-750/',
    #                             name = 'val_new',
    #                               advancement_th=0.45,
    #                          skip_every=12,
    #                          translation=[5, 5],
    #                          time_window=750)
    make_and_run_chain(config)

