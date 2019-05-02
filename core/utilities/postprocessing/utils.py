import dateutil

import cv2

import skimage
import skimage.io
import skimage.feature
import skimage.novice
import skimage.transform
import math

import pandas as pd
import numpy as np

from os import path
from pypeln import thread as th
from tf.transformations import euler_from_quaternion

def csvs2dfs(files):
    stage = th.map(pd.read_csv, files)
    data = list(stage)
    return data


def dfs2df(dfs):
    df = pd.concat([df for df in dfs])
    return df


def filename2map(filename):
    dirs, _ = path.split(filename)
    map_name = path.basename(dirs)

    return map_name


def df_convert_date2timestamp(df):
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
    df['ros_time'] = df.index
    df['timestamp'] = df['ros_time'].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
    df['timestamp'] -= df['timestamp'][0]
    df = df.set_index(df['timestamp'])
    # reset back the index to the time

    return df


def get_pose(row):
    return (row['pose__pose_position_x'], row['pose__pose_position_y'])

def df_convert_quaterion2euler(df):
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

    df[['pose__pose_e_orientation_x', 'pose__pose_e_orientation_y', 'pose__pose_e_orientation_z']] = df.apply(convert,
                                                                                                              axis=1)

    return df


def read_image(heightmap_png):
    """
    Read a given image and convert it to gray scale, then scale to [0,1]
    :param heightmap_png:
    :return:
    """
    hm = cv2.imread(heightmap_png)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    return hm


def to_hm_coordinates(row, hm, res, tr=[0, 0]):
    """
    Convert robot x and y with respect to the heightmap.
    :param row:
    :param hm:
    :param res:
    :param tr:
    :return:
    """
    x, y = get_pose(row)
    x_max, y_max = hm.shape[0] * res, hm.shape[1] * res
    x_min, y_min = tr

    xs = x + x_min
    ys = -y + y_min

    return pd.Series([xs / res, ys / res])


# def hmpatch(hm, x, y, alpha, edge, scale=1):
#     """
#     Cutout a patch from the image, centered on (x,y), rotated by alpha
#     degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
#     with a specified edge size (in pixels) and scale (relative).
#     :param hm:
#     :param x:
#     :param y:
#     :param alpha:
#     :param edge:
#     :param scale:
#     :return:
#     """
#     tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
#     tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
#     tf3 = skimage.transform.SimilarityTransform(scale=scale)
#     tf4 = skimage.transform.SimilarityTransform(translation=[+edge / 2, +edge / 2])
#     tf = (tf1 + (tf2 + (tf3 + tf4))).inverse
#
#     corners = tf(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * edge)
#     patch = skimage.transform.warp(hm, tf, output_shape=(edge, edge), mode="edge")
#     return patch, corners

def hmpatch(hm, x, y, alpha, edge, scale=1, offset=(0,0)):
    """
    Cutout a patch from the image, centered on (x,y), rotated by alpha
    degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
    with a specified edge size (in pixels) and scale (relative).
    :param hm:
    :param x:
    :param y:
    :param alpha:
    :param edge:
    :param scale:
    :return:
    """
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+(edge / 2) + offset[0],
                                                             +(edge / 2) + offset[1]])
    tf = (tf1 + (tf2 + (tf3 + tf4))).inverse

    corners = tf(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * edge)
    patch = skimage.transform.warp(hm, tf, output_shape=(edge + offset[1], edge + offset[0]), mode="edge")
    return patch, corners


def krock_hmpatch(hm, x, y, alpha, max_advancement, res=0.02, debug=False):
    missing_krock_body = KrockDims.KROCK_SIZE - KrockDims.HEAD_OFFSET
    patch_size = (max_advancement + KrockDims.HEAD_OFFSET) / res * 2

    if debug:
        print('[INFO] patch_size = {}'.format(patch_size))
        print('[INFO] missing_krock_body = {}'.format(missing_krock_body))

    offset = (math.ceil((missing_krock_body - max_advancement) / res), 0)
    if debug: print('[INFO] offset = {}'.format(offset))

    return  hmpatch(hm, x, y, alpha, math.ceil(patch_size), offset=offset)

class KrockDims:
    KROCK_SIZE = 0.85
    HEAD_OFFSET = 0.14


class PatchExtractStrategy():
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, hm, x, y, alpha, scale=1, offset=(0,0)):
        """
        Cutout a patch from the image, centered on (x,y), rotated by alpha
        degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
        with a specified edge size (in pixels) and scale (relative).
        :param hm:
        :param x:
        :param y:
        :param alpha:
        :param edge:
        :param scale:
        :return:
        """
        edge = self.patch_size
        tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
        tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
        tf3 = skimage.transform.SimilarityTransform(scale=scale)
        tf4 = skimage.transform.SimilarityTransform(translation=[+(edge / 2) + offset[0] ,
                                                                 +(edge / 2) + offset[1]])
        tf = (tf1 + (tf2 + (tf3 + tf4))).inverse

        corners = tf(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * edge)
        patch = skimage.transform.warp(hm, tf, output_shape=(edge + offset[1], edge + offset[0]), mode="edge")
        return patch, corners

class KrockPatchExtractStrategy(PatchExtractStrategy):
    def __init__(self, max_advancement,  debug=False):
        self.max_advancement = max_advancement

        super().__init__(None)

    @staticmethod
    def patch_shape(max_advancement, res=0.02):
        missing_krock_body =  math.ceil(KrockDims.KROCK_SIZE / res)
        max_advancement = math.ceil(max_advancement / res)
        shape = (max_advancement *2, missing_krock_body + max_advancement)

        return shape

    def __call__(self, hm, x, y, alpha, res=0.02, debug=False):
        max_advancement = self.max_advancement
        missing_krock_body = KrockDims.KROCK_SIZE - KrockDims.HEAD_OFFSET
        # patch_size = (missing_krock_body + KrockDims.HEAD_OFFSET + max_advancement) / res
        patch_size = (max_advancement) / res * 2

        if debug:
            print('[INFO] patch_size = {}'.format(patch_size))
            print('[INFO] missing_krock_body = {}'.format(missing_krock_body))

        offset = (math.ceil((missing_krock_body - max_advancement) / res), 0)
        if debug: print('[INFO] offset = {}'.format(offset))

        self.patch_size = math.ceil(patch_size)

        edge = self.patch_size
        tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
        tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
        tf3 = skimage.transform.SimilarityTransform(scale=1)
        tf4 = skimage.transform.SimilarityTransform(translation=[+(edge / 2) + offset[0],
                                                                 +(edge / 2) + offset[1]])
        tf = (tf1 + (tf2 + (tf3 + tf4))).inverse

        corners = tf(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * edge)
        patch = skimage.transform.warp(hm, tf, output_shape=(edge + offset[1], edge + offset[0] + (KrockDims.HEAD_OFFSET // res)), mode="edge")
                
        return patch, corners


class KrockPatchExtractStrategyNumpy(KrockPatchExtractStrategy):
    @staticmethod
    def fill(im, x, y, value,  max_advancement, res=0.02):
        missing_krock_body = KrockDims.KROCK_SIZE - KrockDims.HEAD_OFFSET
        patch_size = (max_advancement) / res * 2
        offset = (math.ceil((missing_krock_body - max_advancement) / res), 0)

        patch_size = math.ceil(patch_size)

        half = (patch_size // 2)
        x,y = int(x), int(y)

        im[x - half:x + half, int(y - half - offset[0] -  (KrockDims.HEAD_OFFSET // res)):y + half] += value

        return im

    def __call__(self, hm, x, y, alpha, res=0.02, debug=False):
        max_advancement = self.max_advancement
        missing_krock_body = KrockDims.KROCK_SIZE - KrockDims.HEAD_OFFSET
        patch_size = (max_advancement) / res * 2
        offset = (math.ceil((missing_krock_body - max_advancement) / res), 0)

        patch_size = math.ceil(patch_size)

        half = (patch_size // 2)
        x,y = int(x), int(y)

        patch = hm[x - half:x + half,
                int(y - half - offset[0] -  (KrockDims.HEAD_OFFSET // res)):y + half ]

        return patch, ()