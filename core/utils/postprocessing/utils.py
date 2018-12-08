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

from tf.transformations import euler_from_quaternion
# skelearn
import sklearn.pipeline
import sklearn.dummy
import sklearn.preprocessing
import sklearn.metrics.regression
from sklearn.metrics import auc, roc_curve
import skimage.transform
# Each csv contains 20s of simulation from gazebo: pioneer on heightmap
time_window = 75  # Ideally: each row in the csv is about 0.01 segs, so 50 is about .50 s in the future, 100 1. s
# However: sometimes due to the simulation load, each row is about 0.02 segs.
# A preferred way of counting would be in time instead of windows.

#
patch_size = 60  # for extracting patchs from the heightmap for training, eval and testing datasets
# Pioneer is about 50cm long x 47cm wide x 40cm height
# heightmaps are render with a 2cm per pixel resolution; 513x513px --> 10x10m ~1m maxheight

patch_size_training = patch_size  # with 30 we resize to this for training the cnn, it will allow us to deal with small maps

#
advancement_th = 0.10  # threshold in meters use to generate the training dataset, i.e. when a patch is traversed
# this has to be set according to the pioneer velocity and its ideal displacement (flat land)
# .15m/s is the current linear velocity (assuming a forward no steering control)
# ergo, ideal_displacement = .15m/s x (timewindow in seconds)

debug = 0  # debug level for extra logging and intermedia plots, 0 no debuggin -- 3 shows everyhing
           #

multiprocessing = False # if True, we use jobs to generate dataset/calculate the traversability/ plot over a full map

multiprocessing_hm = np.zeros((100,100)) # a temporal way to initialize a shared image

sim_hm_mx_x = 5.0  # heightmap dimmensions (m) used in the simulation for generating training data
sim_hm_mx_y = 5.0  # this will help to pass from sim coordinates to screen coordinates when generating datasets
                   # usually is a 10m by 10m map, so from -5 to 5


height_scale_factor = 1.0 # for learning heightmaps it was 0 - 1.0; if heighmaps are higher, change accordingly

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
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: dateutil.parser.parse(x).timestamp())
    df[df.columns[0]] -= min(df[df.columns[0]])

    return df

def df_convert_quaterion2euler(df):

    def convert(row):
        quaternion = [row['pose__pose_orientation_w'],
                      row['pose__pose_orientation_x'],
                      row['pose__pose_orientation_y'],
                      row['pose__pose_orientation_z']]


        euler = euler_from_quaternion(quaternion)

        return pd.Series([*euler])

    df[['pose__pose_e_orientation_x', 'pose__pose_e_orientation_y', 'pose__pose_e_orientation_z']] = df.apply(convert, axis=1)

    return df


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
    hm = hm * height_scale_factor #scaled to proper factor (mostly for testing, for training is 1.0)
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

def show(sample,hm):
    O_W_KEY = 'pose__pose_orientation_w'

    patch=hmpatch(hm,sample["hm_x"],sample["hm_y"],np.rad2deg(sample[O_W_KEY]),patch_size,scale=1)[0] # make sure to extract the patch from the correct heightmap
    patch=patch-patch[patch.shape[0]//2,patch.shape[1]//2]
    fig,ax1=plt.subplots(figsize=(7,7))
    ax1.imshow(patch/height_scale_factor,cmap="coolwarm",vmin=-0.1,vmax=+0.1)
    ax1.set_title("advancement: {:.4f}".format(sample["advancement"]))
    plt.show()
    plt.close(fig)


def generate_single_dataset_cnn(df, heightmap_png):
    hm = read_image(heightmap_png)

    df = df.set_index(df.columns[0])
    df.columns = df.columns.map(str.strip)  # strip spaces

    P_X_KEY = 'pose__pose_position_x'
    P_Y_KEY = 'pose__pose_position_y'

    O_W_KEY = 'pose__pose_e_orientation_z'

    if debug > 1:
        plt.figure()
        df.plot.scatter(x=P_X_KEY, y=P_Y_KEY)
        plt.show()

    # % Convert xy to hm coords
    df["hm_x"] = df.apply(
        lambda r: toScreenFrame(r[P_X_KEY], r[P_Y_KEY], sim_hm_mx_x, -sim_hm_mx_x, sim_hm_mx_y, -sim_hm_mx_y)[0] *
                  hm.shape[1], axis=1)
    df["hm_y"] = df.apply(
        lambda r: toScreenFrame(r[P_X_KEY], r[P_Y_KEY], sim_hm_mx_x, -sim_hm_mx_x, sim_hm_mx_y, -sim_hm_mx_y)[1] *
                  hm.shape[0], axis=1)

    # % Plot trajectory
    if debug > 0:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(hm / height_scale_factor)
        ax.plot(df["hm_x"], df["hm_y"], '-y')
        ax.plot(df["hm_x"].iloc[0], df["hm_y"].iloc[0], 'oy')
        plt.show()

    # % Plot angles
    # import numpy as np
    if debug > 1:
        plt.figure()
        np.rad2deg(df[O_W_KEY]).plot()
        plt.show()

    # %
    # Unit vector of robot orientation
    df["S_oX"] = np.cos(df[O_W_KEY])
    df["S_oY"] = np.sin(df[O_W_KEY])
    assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))

    # dX, dY, distance at 10 timesteps in the future
    dt = time_window
    df["S_dX"] = df.rolling(window=(dt + 1))[P_X_KEY].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
    df["S_dY"] = df.rolling(window=(dt + 1))[P_Y_KEY].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
    df["S_d"] = np.linalg.norm(df[["S_dX", "S_dY"]], axis=1)
    if debug > 1:
        plt.figure()
        df["S_d"].plot()

    # % Project dX, dY on current direction
    df["advancement"] = np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

    # set the label using a threshold value
    df["label"] = df["advancement"] > advancement_th

    # % Filter data
    # skip the first two seconds and any row with nans (i.e. end of the dataset)
    dff = df.loc[df.index >= 2].dropna()
    dff = dff.loc[dff.index <= 18].dropna()  # drop also the last two seconds (if run is 20s, < 18)

    # drop the frames where the robot is upside down (orientation alpha angle [euler's angles]) to avoid false positives
    # dff = dff.loc[dff.S_RCO_A >= -2.0].dropna()
    # dff = dff.loc[dff.S_RCO_A <= 2.0].dropna()
    #
    # dff = dff.loc[dff.S_RCO_B >= -2.0].dropna()
    # dff = dff.loc[dff.S_RCO_B <= 2.0].dropna()

    # % Visualize the data
    if debug > 2:
        print("Slow")
        for i, sample in dff.sort_values("advancement").head(20).iterrows():
            show(sample, hm)

        print("Fast")
        for i, sample in dff.sort_values("advancement").tail(20).iterrows():
            show(sample, hm)

    return dff, hm
