import matplotlib
import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

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


def create2dtrace(world):

    fig = plt.figure()

    plt.rcParams['figure.figsize'] = [8, 5]
    plt.imshow(world * 255)
    plt.show()


def create3dtrace(world, X, Y, map_name):

    Xm, Ym = np.meshgrid(X, Y)
    Z = cv2.imread('/home/francesco/Documents/Master-Thesis/core/maps/{}.png'.format(map_name))
    Z = cv2.cvtColor(Z, cv2.COLOR_BGR2GRAY)
    Z = Z / 255
    Z = cv2.resize(Z, Xm.shape, cv2.INTER_CUBIC)

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