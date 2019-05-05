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
from utilities.postprocessing.utils import KrockPatchExtractStrategy, PatchExtractStrategy

class StoreDataframeKeepingSameName():
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def __call__(self, data):
        df, hm, filename = data
        df.to_csv(self.out_dir + filename + '.csv')
        return  df, hm, filename


class Bags2Dataframe(Handler):
    """
    This class loads the bags file and converted them to Pandas' dataframe. In addition,
    it opens each map for each file and add it to the return tuple.
    """

    def __init__(self, base_dir, out_dir=None):
        self.base_dir = base_dir

    def __call__(self, filename):
        df = rosbag_pandas.bag_to_dataframe(self.base_dir + filename + '.bag')
        return df, None, filename

class ReadDataframeFilenameAndHm():
    def __init__(self, base_dir, hm_dir):
        self.base_dir = base_dir
        self.hm_dir = hm_dir

    def __call__(self, data):
        idx, row = data
        df, hm = open_df_and_hm_from_meta_row(row, self.base_dir, self.hm_dir)
        df['height'] = row['height']
        return df, hm, row['filename']

class ParseDataframe(Handler):
    def convert_date2timestamp(self, df):
        """
        Convert the index column of the given dataframe which contains the converted bag time
        to a time expressed in second starting from the beginning. For example,
        1549572176.951121,
        1549572176.8528721

        becomes

        0.0,
        0.1df['timestamp']

        :param df:
        :return:
        """
        df.index = df[df.columns[0]]
        df['ros_time'] = df.index
        try:
            df['timestamp'] = df['ros_time'].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
            df['timestamp'] -= df['timestamp'].iloc[0]
            df = df.set_index(df['timestamp'])
        except Exception:
            print('[INFO] something exploded while converting the ros time.')
        return df

    def convert_quaterion2euler(self, df, *args, **kwargs):
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

    def __call__(self, data):
        df, hm, filename = data

        df = self.convert_date2timestamp(df)
        df = self.convert_quaterion2euler(df)
        df = self.extract_cos_sin(df)
        return df, hm, filename


class AddAdvancement(Handler):
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, data):
        """
        Project the distance x and y computed using a rolling window
        into the current line to compute the advancement
        :param df:
        :param dt:
        :return:
        """
        df, hm, filename = data

        if len(df) > 0:

            # look dt in the future and compute the distance for booth axis
            df["S_dX"] = df.rolling(window=(self.dt + 1))['pose__pose_position_x'].apply(lambda x: x[-1] - x[0],
                                                                                         raw=True).shift(
                -self.dt)
            df["S_dY"] = df.rolling(window=(self.dt + 1))['pose__pose_position_y'].apply(lambda x: x[-1] - x[0],
                                                                                         raw=True).shift(
                -self.dt)
            # project x and y in the current line and compute the advancement
            df["advancement"] = np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

            df = df.dropna()

        return df, hm, filename


class CleanDataframe(Handler):
    def __init__(self, lower_bound, offset=0):
        self.lower_bound = lower_bound
        self.offset = offset

    def __call__(self, data):
        offset = self.offset

        df, hm, filename = data
        #       drop first second (spawn time)
        df = df.loc[df.index >= self.lower_bound]
        # robot upside down
        df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
        df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()
        df = df.dropna()

        index = df[(df['hm_y'] > (hm.shape[0] - offset)) | (df['hm_y'] < offset)
                   | (df['hm_x'] > (hm.shape[1] - offset)) | (df['hm_x'] < offset)
                   ].index

        # print('removing {} outliers'.format(len(index)))
        # if there are some outliers, we remove all the rows after the first one
        if len(index) > 0:
            idx = index[0]
            df = df.loc[0:idx]

        return df, hm, filename


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
        df, hm, filename = data
        df[['hm_x', 'hm_y']] = df.apply(
            lambda x: self.to_hm_coordinates(x, hm, self.resolution, self.translation),
            axis=1)
        return df, hm, filename

def drop_uselesss_columns(data):
    df, hm, filename = data

    df = df.drop(['pose__pose_orientation_y',
                  'pose__pose_orientation_x',
                  'pose__pose_orientation_z',
                  'pose__pose_orientation_w',
                  'timestamp',
                  'ros_time'], axis=1)

    return df, hm, filename

class ReadHm(Handler):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def __call__(self, name):
        return read_image(self.base_dir + name + '.png')


def open_df_and_hm_from_meta_row(row, base_dir, hm_dir):
    filename, map = row['filename'], row['map']
    df = pd.read_csv(base_dir + '/' + filename + '.csv')

    hm = cv2.imread(hm_dir + '/' + map + '.png')
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

    return df, hm

class ReadDataframeAndStoreName():
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def __call__(self, path):
        df = pd.read_csv(self.base_dir + path + '.csv')
        return df

class ExtractPatches():
    def __init__(self, patch_extract_stategy):
        self.patch_extract_stategy = patch_extract_stategy

    def __call__(self, data):
        df, hm, filename = data

        patches = []

        for (idx, row) in df.iterrows():
            patch = self.patch_extract_stategy(hm, row["hm_x"], row["hm_y"], np.rad2deg(row['pose__pose_e_orientation_z']))[0]
            patches.append(patch)

        return df, patches, filename

class StorePatches():
    def __init__(self, out_dir, meta_df_out_dir):
        self.out_dir = out_dir
        self.meta_df_out_dir = meta_df_out_dir

    def __call__(self, data):
        df, patches, filename = data

        paths = []
        for idx, patch in enumerate(patches):
            patch_file_name = '{}-{}.png'.format(filename, idx)
            path = '{}/{}'.format(self.out_dir, patch_file_name)
            patch = patch * 255
            patch = patch.astype(np.uint8)
            cv2.imwrite(path, patch)
            paths.append(patch_file_name)

        df['images'] = paths

        df.to_csv('{}/{}.csv'.format(self.meta_df_out_dir, filename))
        del patches # free up memory

        return df, filename


