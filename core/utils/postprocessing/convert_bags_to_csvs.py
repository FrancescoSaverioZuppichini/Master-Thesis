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

WORKERS = 16

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
    return (df, map_name, file_name)

def df2csv(data):
    file_name, df = data
    df.to_csv(file_name)

def bags2dfs(files):
    stage = th.map(bag2df, files, workers=WORKERS)
    data = list(stage)
    return data

def dfs2df(dfs):
    df = pd.concat([df for df in dfs])
    return df

def files2bags(files):
    for file in files:
        bag = rosbag.Bag(file)
        yield bag

def bags2csv(files):
    data = bags2dfs(files)

    file_names = [(path.splitext(file_name)[0] + '.csv', df)  for df, _, file_name in data]

    stage = th.map(df2csv, file_names, workers=WORKERS)

    csv = list(stage)

    return csv

BAG_FOLDER = '/home/francesco/Desktop/carino/vaevictis/data/'
files = glob.glob(BAG_FOLDER + '/*/**.bag')

start = time.time()
bags2csv(files)
finish = time.time()
print('Run in {:.4f}'.format(finish - start))