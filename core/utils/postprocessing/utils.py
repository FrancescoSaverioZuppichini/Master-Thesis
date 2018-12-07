import os
import glob
import rosbag
import rosbag_pandas
import numpy as np
import pandas as pd
import timeit
import cv2
import time

from os import path
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from pypeln import thread as th

def filename2map(filename):
    dirs, _ = path.split(filename)
    map_name = path.basename(dirs)

    return map_name

def bag2df(file_name):
    df = None
    try:
        df = rosbag_pandas.bag_to_dataframe(file_name)
    except TypeError:
        print(file_name)
    map_name = filename2map(file_name)
    return (map_name, df)

def bags2dfs(files, *args, **kwargs):
    stage = th.map(bag2df, files, *args, **kwargs)
    data = list(stage)
    return data

def dfs2df(dfs):
    df = pd.concat([df for df in dfs])
    return df

def files2bags(files):
    for file in files:
        bag = rosbag.Bag(file)
        yield bag


# def bags2csv(files):
#     data = bags2dfs(files)
#
#     print(data)