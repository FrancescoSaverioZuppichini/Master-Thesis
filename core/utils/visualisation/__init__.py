import matplotlib
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from os import path

from utils.postprocessing.utils import hmpatch

MAPS_DIR = '/home/francesco/Documents/Master-Thesis/core/maps/'

def file_name2hm(file_name):
    map_name = path.basename(path.dirname(file_name))
    hm = cv2.imread('{}/{}.png'.format(MAPS_DIR, map_name))
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

    return hm

def create_trace_world(df, res=0.1):
    x = df.pose__pose_position_x
    y = df.pose__pose_position_y

    X = np.arange(-5, 5, res)
    Y = np.arange(-5, 5, res)

    world = np.zeros((X.shape[0], Y.shape[0]))

    for x_, y_ in zip(x, y):
        xi = np.digitize(x_, X) - 1
        yi = np.digitize(y_, Y) - 1
        world[xi, yi] += 1

    world /= len(x)
    world = world.T

    return world, X, Y

def show_trace(df, hm):
    fig, ax = plt.subplots()

    ax.imshow(hm)
    ax.plot(df.hm_x,  df.hm_y, '--', linewidth=1, color='firebrick')

def show_naked_trace(df):
    fig = plt.figure()
    plt.plot(df.hm_x,  df.hm_y, '--', linewidth=1, color='firebrick')

def show_naked_traces(dfs):
    fig = plt.figure()
    for df in dfs:
        plt.plot(df.hm_x,  df.hm_y, '--', linewidth=1, color='firebrick')
def show_traces(dfs, hm):
    fig, ax = plt.subplots()

    ax.imshow(hm)
    for df in dfs:
        ax.plot(df.hm_x, df.hm_y, '--', linewidth=1, color='firebrick')


def create3dtrace(world, X, Y, map_name):
    Xm, Ym = np.meshgrid(X, Y)
    Z = cv2.imread('{}/{}.png'.format(MAPS_DIR, map_name))
    Z = cv2.cvtColor(Z, cv2.COLOR_BGR2GRAY)
    Z = Z / 255
    Z = cv2.resize(Z, Xm.shape)

    plt.imshow(Z)
    # fourth dimention - colormap
    # create colormap according to x-value (can use any 50x50 array)
    color_dimension = world  # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(Xm, Ym, Z, facecolors=fcolors, linewidth=0.1)
    plt.show()

def show_advancement(df, hm, config):
    print("Slow")
    for i, sample in df.sort_values("advancement").head(20).iterrows():
        show(sample, hm, config)

    print("Fast")
    for i, sample in df.sort_values("advancement").tail(20).iterrows():
        show(sample, hm, config)

def show(sample,hm, config):
    O_W_KEY = 'pose__pose_e_orientation_z'

    patch=hmpatch(hm,sample["hm_x"],sample["hm_y"],np.rad2deg(sample[O_W_KEY]),config.PATCH_SIZE,scale=1)[0] # make sure to extract the patch from the correct heightmap
    patch=patch-patch[patch.shape[0]//2,patch.shape[1]//2]
    fig = plt.figure(figsize=(16,16))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    ax1.imshow(patch/config.HEIGHT_SCALE_FACTOR)
    ax1.set_title("advancement: {:.4f}, x={:.0f}, y={:.0f}".format(sample["advancement"], sample["hm_x"],sample["hm_y"]))

    ax2.imshow(hm)
    rect = patches.Rectangle((sample["hm_x"] - config.PATCH_SIZE/2,sample["hm_y"] - config.PATCH_SIZE/2), config.PATCH_SIZE, config.PATCH_SIZE, linewidth=1, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    plt.show()

def show_trajectory(samples,hm):
    cmap = plt.cm.get_cmap('Spectral')
    O_W_KEY = 'pose__pose_e_orientation_z'
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(hm)

    for i, sample in samples:
        if i % 100 == 0:
            rect = patches.Rectangle((sample["hm_x"],sample["hm_y"]), 10, 10, linewidth=1, edgecolor='none',
                                     facecolor=cmap(sample['advancement'] / 0.16),
                                     alpha=0.6)
            ax.add_patch(rect)

    plt.show()