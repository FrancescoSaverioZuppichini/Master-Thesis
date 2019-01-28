import glob
import rosbag
import rosbag_pandas
import time
import os
from os import path
from pypeln import thread as th
from utils import *
from config import Config

def bag2df(file_name):
    df = None
    try:
        df = rosbag_pandas.bag_to_dataframe(file_name)
    except TypeError:
        print(file_name)
    map_name = filename2map(file_name)
    return (df, map_name, file_name)

def df2csv(data):
    file_path, df = data
    os.makedirs(path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)
    return df

def bags2dfs(files):
    stage = th.map(bag2df, files, workers=Config.WORKERS)
    # data = list(stage)
    return stage

def files2bags(files):
    for file in files:
        bag = rosbag.Bag(file)
        yield bag

def bags2csvs(files):
    stage = bags2dfs(files)

    return stage
