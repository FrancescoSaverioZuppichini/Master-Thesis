import os
import rosbag
import pandas as pd
import numpy as np
import rosbag_pandas
import os
from os import path
import glob
import cv2
import dateutil

from config import Config
from utils import *
from pypeln import thread as th

class PostProcessingConfig():
    def __init__(self,  base_dir, maps_folder, patch_size, advancement_th, time_window, skip_every, translation, resolution=0.2, scale=1, n_workers=16,
                 bags_dir=None, csv_dir=None, out_dir=None, verbose=True, patches=True):
        self.base_dir, self.maps_folder, self.bags_dir, self.csv_dir, self.out_dir = base_dir, maps_folder, bags_dir, csv_dir, out_dir
        self.make_dirs()

        self.patch_size, self.advancement_th, self.time_window = patch_size, advancement_th, time_window
        self.scale, self.skip_every =  scale, skip_every
        self.resolution, self.translation = resolution, translation
        self.n_workers = n_workers

    def make_dirs(self):
        if self.bags_dir is None: self.bags_dir  = path.normpath(self.base_dir + '/bags/')
        if self.csv_dir is None: self.csv_dir = path.normpath(self.base_dir + '/csvs/')
        if self.out_dir is None: self.out_dir  = path.normpath(self.base_dir + '/outs/')


    @classmethod
    def from_config(cls, config):
        return cls(config.BASE_DIR, config. MAPS_FOLDER, config.PATCH_SIZE, config.ADVANCEMENT_TH, config.TIME_WINDOW,
                   config.SKIP_EVERY, config.TRANSLATION)


class PostProcessing():
    def __init__(self, config: PostProcessingConfig):
        self.config = config

    def __call__(self, stage):
        pass

class BagsPostProcessing(PostProcessing):

    def bag2df(self, file_name):
        df = rosbag_pandas.bag_to_dataframe(file_name)

        map_name = filename2map(file_name)

        return (df, map_name, file_name)

    def __call__(self, bags):

        stage = th.map(self.bag2df, bags, workers=self.config.n_workers)
        return stage



class DataFramePostProcessing(PostProcessing):

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
        df = df.reset_index()  # we need to drop the index since the time column is used for it
        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
        df['timestamp'] = df[df.columns[0]]
        df[df.columns[0]] -= min(df[df.columns[0]])
        df = df.set_index(df.columns[0])  # reset back the index to the time

        return df


    def df_add_hm_coords(self, df, hm):
        """
        Decorate the current dataframe with the robot's position
        in the heightmap. The x and y coordinates will be stored in 'hm_x' and 'hm_y' columns.
        :param df: current dataframe
        :param hm: heightmap representing the map
        :return: decorated dataframe
        """

        df[['hm_x', 'hm_y']] = df.apply(lambda x: to_hm_coordinates(x, hm, self.config.resolution, self.config.translation),
                                        axis=1)
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
        df["S_oX"] = np.cos(df['pose__pose_e_orientation_z'].values)
        df["S_oY"] = np.sin(df['pose__pose_e_orientation_z'].values)

        assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))
        # look dt in the future and compute the distance for booth axis
        df["S_dX"] = df.rolling(window=(dt + 1))['pose__pose_position_x'].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
        df["S_dY"] = df.rolling(window=(dt + 1))['pose__pose_position_y'].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
        # compute euclidean distance
        df["S_d"] = np.linalg.norm(df[["S_dX", "S_dY"]], axis=1)
        # project x and y in the current line and compute the advancement
        df["advancement"] = np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

        return df


    def df_clean_by_dropping(self, df, max_x, max_y):
        """
        Clean the given dataframe by dropping the rows
        - with time stamp < 2 and > 18 seconds
        - where the robot is upside down
        :param df:
        :param max_x:
        :param max_y:
        :return:
        """
        df = df.loc[df.index >= 1].dropna()
        df = df.loc[df.index <= 19].dropna()

        # robot upside down
        df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()

        # TODO we can use hm_x and hm_y instead!
        # df = df.loc[(df[P_Y_KEY] < max_x - 5) & (df[P_Y_KEY] > -5.1)].dropna()
        # df = df.loc[(df[P_Y_KEY] < max_y - 5) & (df[P_Y_KEY] > -5.1)].dropna()

        return df


    def df_clean_by_removing_outliers(self, df, hm):
        offset = 10

        index = df[(df['hm_y'] > (hm.shape[0] - offset)) | (df['hm_y'] < offset)
                   | (df['hm_x'] > (hm.shape[1] - offset)) | (df['hm_x'] < offset)
                   ].index

        print('removing {} outliers'.format(len(index)))

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

    def dfs2patches(self, data):
        stage = th.map(self.traversability_df2patches, data, workers=self.config.n_workers)
        data = list(stage)
        return data




    def df2traversability_df(self, data):
        df, map_name, file_path = data

        def make_path(file_path):
            splitted = file_path.split('/')
            map_name, file_name = splitted[-2], splitted[-1]
            return path.normpath('{}/{}/{}'.format(self.config.csv_dir, map_name, path.splitext(file_name)[0] + '.csv'))

        map_name = filename2map(file_path)
        map_path = '{}/{}.png'.format(Config.MAPS_FOLDER, map_name)
        hm = read_image(map_path)
        file_path = make_path(file_path)

        if path.isfile(file_path):
            print('file exist, loading...')
            df = pd.read_csv(file_path)

        else:
            df = df_convert_date2timestamp(df)
            df = df_convert_quaterion2euler(df)
            df = self.df_clean_by_dropping(df, hm.shape[0] * self.config.resolution, hm.shape[1] * self.config.resolution)

            if len(df) > 0:
                df = self.df_add_hm_coords(df, hm)
                df = self.df_clean_by_removing_outliers(df, hm)
                df = self.df_add_advancement(df, Config.TIME_WINDOW)
                df = self.df_add_label(df, Config.ADVANCEMENT_TH)

                # TODO add flag to decide if store the csv or not
                os.makedirs(path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path)
            else:
                print('{} contains 0 rows, dropping...'.format(file_path))

        return df, hm, file_path

    def __call__(self, data):
        stage = th.map(self.df2traversability_df, data, workers=self.config.n_workers)
        return stage

class DataFrame2PatchesPostProcessing(PostProcessing):

    def traversability_df2patches(self, data):
        """
        Given a dataframe, and heightmap and the file path generate the patches
        every `Config.SKIP_EVERY` rows. This is done due to the big rate that we used to
        store the data, around 250hz. A `Config.SKIP_EVERY=12` is equal to a rate of 20.
        :param data:
        :return:
        """
        df, hm, file_path = data
        dirs, name = path.split(file_path)
        name, _ = os.path.splitext(name)

        out_dir = self.config.out_dir

        os.makedirs(out_dir + '/True', exist_ok=True)
        os.makedirs(out_dir + '/False', exist_ok=True)

        # reset the index to int so we can take only on row every Config.SKIP_EVERY
        # since the stored rate was really high, 250hz, we will end up with lots of almost
        # identical patches
        df = df.reset_index()
        df = df.loc[list(range(0, len(df), Config.SKIP_EVERY)), :]
        df = df.set_index(df.columns[0])

        for idx, (i, row) in enumerate(df.iterrows()):
            patch = hmpatch(hm, row["hm_x"], row["hm_y"], np.rad2deg(row['pose__pose_e_orientation_z']), self.config.patch_size, scale=1)[0]
            patch = (patch * 255).astype(np.uint8)
            cv2.imwrite('{}/{}/{}.png'.format(out_dir, row['label'], row['timestamp']), patch)

        return data

    def __call__(self, data):
        stage = th.map(self.traversability_df2patches, data, workers=self.config.n_workers)

        return stage

# # post = PostProcessing(patch_size=100, skip_every=12, time_window=125, advancement_th=0.12, resolution=0.02, translation=[5,5])
# #
# # post('./test')
# post_config = PostProcessingConfig.from_config(Config)

post_config = PostProcessingConfig(base_dir='./test/', patch_size=100, time_window=125, advancement_th=0.09,
                                   resolution=0.02, translation=[5,5], maps_folder='/home/francesco/Documents/Master-Thesis/core/maps/test/', skip_every=12)
print(post_config.__dict__)


bags_post = BagsPostProcessing(post_config)
df_post = DataFramePostProcessing(post_config)
df2patches_post = DataFrame2PatchesPostProcessing(post_config)

bags = glob.glob('./test/bags/**/*.bag')

stage = bags_post(bags)
print(list(df2patches_post(df_post(bags_post(bags)))))