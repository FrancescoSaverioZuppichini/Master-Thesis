import skimage
import skimage.io
import skimage.feature
import skimage.novice
import matplotlib.pyplot as plt
import pandas as pd
# numpy
import numpy as np
import math
from os import path

import dateutil
from pypeln import thread as th
try:
    from .config import Config
except ImportError:
    from config import Config

from tf.transformations import euler_from_quaternion
# skelearn
import sklearn.pipeline
import sklearn.dummy
import sklearn.preprocessing
import sklearn.metrics.regression
from sklearn.metrics import auc, roc_curve
import skimage.transform

def file2df_map(file):
    df = pd.read_csv(file)
    map_name = filename2map(file)
    map = read_image(Config.MAPS_FOLDER + map_name + '.png')
    return df, map

def files2dfs_maps(files):
    stage = th.map(file2df_map, files)
    data = list(stage)
    return data

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
    df = df.reset_index()

    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: dateutil.parser.parse(str(x)).timestamp())
    df[df.columns[0]] -= min(df[df.columns[0]])

    df = df.set_index(df.columns[0])
    return df

def get_pose(row):
        return (row['pose__pose_position_x'], row['pose__pose_position_y'])

def df_add_dist_velocity(df):

    dists = []
    vels = []

    for i in range(1, len(df)):
        p1 = get_pose(df.iloc[i - 1])
        p2 = get_pose(df.iloc[i])
        t1 = df.iloc[i - 1]['index']
        t2 = df.iloc[i]['index']

        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        vel = dist / (t2 - t1)

        dists.append(dist)
        vels.append(vel)

    dists.append(None)
    vels.append(None)

    df['distance'] = dists
    df['velocity'] = vels

    return df

def df_convert_quaterion2euler(df):

    def convert(row):
        quaternion = [row['pose__pose_orientation_x'],
                      row['pose__pose_orientation_y'],
                      row['pose__pose_orientation_z'],
                      row['pose__pose_orientation_w']]


        euler = euler_from_quaternion(quaternion)

        return pd.Series(euler)

    df[['pose__pose_e_orientation_x', 'pose__pose_e_orientation_y', 'pose__pose_e_orientation_z']] = df.apply(convert, axis=1)

    return df


def plt_trajectory(hm, df):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(hm / height_scale_factor)
        ax.plot(df["hm_x"], df["hm_y"], '-y')
        ax.plot(df["hm_x"].iloc[0], df["hm_y"].iloc[0], 'oy')
        plt.show()

def read_image(heightmap_png):
    # reads an image takint into account the scalling and the bitdepth
    hm = skimage.io.imread(heightmap_png)
    if hm.ndim > 2: #multiple channels
        hm=skimage.color.rgb2gray(hm) #rgb2gray does the averaging and channel reduction
    elif hm.ndim == 2: #already in one channel
        #this is mostly for the images treated in matlab beforehand (one channel + grayscale + 16bit)
        if hm.dtype == 'uint8':
            divided = 255
        if hm.dtype == 'uint16':
            divided = 65535
        hm=hm/divided
    hm = hm * Config.HEIGHT_SCALE_FACTOR #scaled to proper factor (mostly for testing, for training is 1.0)
    return hm

def toScreenFrame (s_x, s_y, x_max, x_min, y_max, y_min):
    # from simulation frame x right, y up, z out of the screen
    # to x right , y down, ignoring z
    xs = s_x + x_max
    ys = -s_y + y_max
    xs = xs/(x_max-x_min)
    ys = ys/(y_max-y_min)
    return xs, ys

def hmpatch(hm,x,y,alpha,edge,scale=1):
    # Cutout a patch from the image, centered on (x,y), rotated by alpha
    # degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
    # with a specified edge size (in pixels) and scale (relative).
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge/2, +edge/2])
    tf=(tf1+(tf2+(tf3+tf4))).inverse
    #corners=tf(np.array([[0,0],[1,0],[1,1],[0,1]])*edge)
    corners=tf(np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])*edge)
    patch = skimage.transform.warp(hm, tf,output_shape=(edge,edge),mode="edge")
    return patch,corners

def hmpatch_only_corners(x,y,alpha,edge,scale=1):
    # Cutout a patch from the image, centered on (x,y), rotated by alpha
    # degrees (0 means bottom in hm remains bottom in patch, 90 means bottom in hm becomes right in patch),
    # with a specified edge size (in pixels) and scale (relative).
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(alpha))
    tf3 = skimage.transform.SimilarityTransform(scale=scale)
    tf4 = skimage.transform.SimilarityTransform(translation=[+edge/2, +edge/2])
    tf=(tf1+(tf2+(tf3+tf4))).inverse
    corners=tf(np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])*edge)
    #patch = skimage.transform.warp(hm, tf,output_shape=(edge,edge),mode="edge")
    return corners

