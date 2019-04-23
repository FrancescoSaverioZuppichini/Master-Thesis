import dateutil
import rosbag_pandas
import cv2
import skimage
import skimage.io
import skimage.feature
import skimage.novice
import skimage.transform

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pypeln import thread as th
from utilities.pipeline import Compose, Handler, Combine, MultiThreadWrapper
from tf.transformations import euler_from_quaternion
from utilities.postprocessing.utils import read_image


class StoreDataframeKeepingSameName():
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def __call__(self, data):
        df, file_name = data
        df.to_csv(self.out_dir + file_name + '.csv')
        return df


class Bags2Dataframe(Handler):
    """
    This class loads the bags file and converted them to Pandas' dataframe. In addition,
    it opens each map for each file and add it to the return tuple.
    """

    def __init__(self, base_dir, out_dir=None):
        self.base_dir = base_dir

    def __call__(self, file_name):
        df = rosbag_pandas.bag_to_dataframe(self.base_dir + file_name + '.bag')
        df.file_name = file_name
        return df


class ParseDataframe(Handler):
    def convert_date2timestamp(self, df):
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
        df.index = df[df.columns[0]]
        df['ros_time'] = df.index
        df['timestamp'] = df['ros_time'].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
        df['timestamp'] -= df['timestamp'][0]
        df = df.set_index(df['timestamp'])

        return df

    def convert_quaterion2euler(self, df):
        """
        Decorate the given dataframe with the euler orientation computed from the existing
        quaternion.
        :param df:
        :return:
        """

        def convert(row):
            quaternion = [row['pose__pose_orientation_x'],
                          row['pose__pose_orientation_y'],
                          row['pose__pose_orientation_z'],
                          row['pose__pose_orientation_w']]

            euler = euler_from_quaternion(quaternion)

            return pd.Series(euler)

        df[['pose__pose_e_orientation_x', 'pose__pose_e_orientation_y', 'pose__pose_e_orientation_z']] = df.apply(
            convert,
            axis=1)

        return df

    def extract_cos_sin(self, df):
        df["S_oX"] = np.cos(df['pose__pose_e_orientation_z'].values)
        df["S_oY"] = np.sin(df['pose__pose_e_orientation_z'].values)

        assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))
        return df

    def __call__(self, df):
        df = self.convert_date2timestamp(df)
        df = self.convert_quaterion2euler(df)
        df = self.extract_cos_sin(df)
        return df


class AddAdvancement(Handler):
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, df):
        """
        Project the distance x and y computed using a rolling window
        into the current line to compute the advancement
        :param df:
        :param dt:
        :return:
        """
        # look dt in the future and compute the distance for booth axis
        df["S_dX"] = df.rolling(window=(self.dt + 1))['pose__pose_position_x'].apply(lambda x: x[-1] - x[0],
                                                                                     raw=True).shift(
            -self.dt)
        df["S_dY"] = df.rolling(window=(self.dt + 1))['pose__pose_position_y'].apply(lambda x: x[-1] - x[0],
                                                                                     raw=True).shift(
            -self.dt)
        # project x and y in the current line and compute the advancement
        df["advancement"] = np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

        return df


class CleanDataframe(Handler):
    def __call__(self, df):
        df = df.dropna()
        #       drop first second (spawn time)
        df = df.loc[df.index >= 1]
        # robot upside down
        df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()

        return df


class AddHMcoordinates(Handler):
    def __init__(self, resolution=0.02, translation=[5, 5]):
        self.resolution = resolution
        self.translation = translation

    def to_hm_coordinates(self, row, hm, res, tr=[0, 0]):
        """
        Convert robot x and y with respect to the heightmap.
        :param row:
        :param hm:
        :param res:
        :param tr:
        :return:
        """
        x, y = (row['pose__pose_position_x'], row['pose__pose_position_y'])
        x_max, y_max = hm.shape[0] * res, hm.shape[1] * res
        x_min, y_min = tr

        xs = x + x_min
        ys = -y + y_min

        return pd.Series([xs / res, ys / res])

    def __call__(self, data):
        """
        Decorate the current dataframe with the robot's position
        in the heightmap. The x and y coordinates will be stored in 'hm_x' and 'hm_y' columns.
        :param df: current dataframe
        :param hm: heightmap representing the map
        :return: decorated dataframe
        """
        df, hm = data
        df[['hm_x', 'hm_y']] = df.apply(
            lambda x: self.to_hm_coordinates(x, hm, self.resolution, self.translation),
            axis=1)
        return df

DATASET_DIR = '/media/francesco/saetta/krock-dataset/'
base_bags_dir = DATASET_DIR + '/train/bags/'
out_csvs_dir = '/media/francesco/saetta/krock-dataset/train/csvs/'

meta = pd.read_csv(DATASET_DIR + 'train/bags/meta.csv')
meta['height'] = 1  # set height to one for now


filename = meta['filename'][0:24]

dfs_from_bags = MultiThreadWrapper(16, Bags2Dataframe(base_bags_dir))(filename)

save_dfs_from_basgs = MultiThreadWrapper(16, StoreDataframeKeepingSameName(out_csvs_dir))
save_dfs_from_basgs(Combine()(dfs_from_bags, filename))


out_parsed_csvs_dir = '/media/francesco/saetta/krock-dataset/train/csvs_parsed/'


parse_df =  MultiThreadWrapper(16, Compose([lambda x: pd.read_csv(out_csvs_dir + x + '.csv'),
                    ParseDataframe()]))

parsed_dfs = parse_df(filename)

save_parsed_dfs = MultiThreadWrapper(16, StoreDataframeKeepingSameName(out_parsed_csvs_dir))
save_parsed_dfs(Combine()(parsed_dfs, filename))