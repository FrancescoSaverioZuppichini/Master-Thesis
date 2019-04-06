import dateutil

import cv2

import skimage
import skimage.io
import skimage.feature
import skimage.novice
import skimage.transform

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
    df = df.reset_index()  # we need to drop the index since the time column is used for it
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
    df['timestamp'] = df[df.columns[0]]
    df[df.columns[0]] -= min(df[df.columns[0]])
    df = df.set_index(df.columns[0])  # reset back the index to the time

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


def hmpatch(hm, x, y, alpha, edge, scale=1):
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
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge / 2, +edge / 2])
    tf = (tf1 + (tf2 + (tf3 + tf4))).inverse

    corners = tf(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * edge)
    patch = skimage.transform.warp(hm, tf, output_shape=(edge, edge), mode="edge")
    return patch, corners
