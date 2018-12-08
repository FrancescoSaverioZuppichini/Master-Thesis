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

def bags2dfs(files):
    stage = th.map(bag2df, files, workers=Config.WORKERS)
    data = list(stage)
    return data

def dfs2df(dfs):
    df = pd.concat([df for df in dfs])
    return df

def files2bags(files):
    for file in files:
        bag = rosbag.Bag(file)
        yield bag

def bags2csvs(files):
    data = bags2dfs(files)

    def make_path(file_path):
        splitted = file_path.split('/')
        map_name, file_name = splitted[-2], splitted[-1]

        return path.normpath('{}/{}/{}'.format(Config.CSV_FOLDER, map_name, path.splitext(file_name)[0] + '.csv'))

    file_patches = [(make_path(file_path), df)  for df, _, file_path in data]

    stage = th.map(df2csv, file_patches, workers=Config.WORKERS)

    data = list(stage)

    return data
