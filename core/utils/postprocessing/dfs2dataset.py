import os
import pandas as pd
import numpy as np

from os import path
from pypeln import thread as th

from utils import *
from config import Config
from pypeln import thread as th
import matplotlib.pyplot as plt
import cv2
import time
from math import floor

P_X_KEY = 'pose__pose_position_x'
P_Y_KEY = 'pose__pose_position_y'
P_Z_KEY = 'pose__pose_position_z'

O_W_E_KEY = 'pose__pose_e_orientation_z'

def df_add_hm_coords(df, hm, sim_hm_mx_x, sim_hm_mx_y):
    # the map in webots is always shitted by 5,5
    df["hm_x"] = df.apply(
        lambda r: toScreenFrame(r[P_X_KEY], r[P_Y_KEY], sim_hm_mx_x, -5, sim_hm_mx_y, -5)[0], axis=1)
    df["hm_y"] = df.apply(
        lambda r: toScreenFrame(r[P_X_KEY], r[P_Y_KEY], sim_hm_mx_x, -5, sim_hm_mx_y, -5)[1], axis=1)

    return df

def df_add_advancement(df, dt):
    # df = df.set_index(df.columns[0])

    df["S_oX"] = np.cos(df[O_W_E_KEY].values)
    df["S_oY"] = np.sin(df[O_W_E_KEY].values)
    assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))

    df["S_dX"] = df.rolling(window=(dt + 1))[P_X_KEY].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
    df["S_dY"] = df.rolling(window=(dt + 1))[P_Y_KEY].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
    df["S_d"] = np.linalg.norm(df[["S_dX", "S_dY"]], axis=1)

    df["advancement"] =  np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

    return df

def df_clean_by_dropping(df, max_x, max_y):
    # skip the first two seconds and any row with nans (i.e. end of the dataset)
    df = df.loc[df.index >= 2].dropna()
    df = df.loc[df.index <= 18].dropna()  # drop also the last two seconds (if run is 20s, < 18)

    # drop the frames where the robot is upside down (orientation alpha angle [euler's angles]) to avoid false positives
    df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()

    df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()
    # TODO not sure if this is okay also for the querry map
    # df = df.loc[(df[P_Y_KEY] <max_x) & (df[P_Y_KEY] > -5.1)].dropna()
    # df = df.loc[(df[P_Y_KEY] < max_y) & (df[P_Y_KEY] > -5.1)].dropna()


    return df

def df_add_label(df, advancement_th):
    df["label"] = df["advancement"] > advancement_th
    return df

def traversability_df2paths(data):
    df, hm, file_path = data
    dirs, name = path.split(file_path)
    name, _ = os.path.splitext(name)

    # out_dir ='{}/{}/{}'.format( Config.IMAGES_DATASET_FOLDER, path.basename(dirs), name)
    out_dir = Config.IMAGES_DATASET_FOLDER

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/True', exist_ok=True)
    os.makedirs(out_dir + '/False', exist_ok=True)

    img_names, img_labels = [], []
    # reset the index to int so we can take only on row every Config.SKIP_EVERY
    # since the stored rate was really high, 250hz, we will end up with lots of almost
    # identical paths
    df = df.reset_index()
    df = df.loc[list(range(0, len(df), Config.SKIP_EVERY)), :]
    df = df.set_index(df.columns[0])
    # for each row,
    for idx, (i, row) in enumerate(df.iterrows()):
        patch = hmpatch(hm,row["hm_x"],row["hm_y"],np.rad2deg(row[O_W_E_KEY]),Config.PATCH_SIZE,scale=1)[0]
        patch = patch-patch[patch.shape[0]//2,patch.shape[1]//2]
        if Config.CENTER_H_PATCH:
            to_hm = lambda x: min(int(x), hm.shape[0] - 1)
            z_hm = hm[to_hm(row["hm_x"])][to_hm(row["hm_y"])]
            patch -= z_hm
        patch = (patch * 255).astype(np.uint8)
        cv2.imwrite('{}/{}/{}-{}.png'.format(out_dir, row['label'], i, time.time()), patch)
        # print('.')

    df_new = pd.DataFrame(data={'name': img_names, 'label': img_labels})

    df_new.to_csv(out_dir + '/meta.csv')
    return df_new

def dfs2paths(data):
    stage = th.map(traversability_df2paths, data, workers=Config.WORKERS)
    data = list(stage)
    return data


def traversability_dfs2paths(data):
    stage = th.map(traversability_df2paths, data, workers=Config.WORKERS)

    return stage

def df2traversability_df(data):
    df, map_name, file_path = data
    map_name = filename2map(file_path)
    map_path = '{}/{}.png'.format(Config.MAPS_FOLDER, map_name)
    hm  = read_image(map_path)
    df = df_convert_date2timestamp(df)
    # TODO place it somewhere (e.g. Config)
    resolution = 0.02
    # df = df_add_dist_velocity(df)
    df = df_convert_quaterion2euler(df)
    df = df_add_hm_coords(df, hm, hm.shape[0] / 100, hm.shape[1] / 100)
    df = df_add_advancement(df, Config.TIME_WINDOW)
    df = df_add_label(df, Config.ADVANCEMENT_TH)
    df = df_clean_by_dropping(df, hm.shape[0] / 100, hm.shape[1] / 100)

    def make_path(file_path):
        splitted = file_path.split('/')
        map_name, file_name = splitted[-2], splitted[-1]

        return path.normpath('{}/{}/{}'.format(Config.CSV_FOLDER, map_name, path.splitext(file_name)[0] + '.csv'))
    # TODO add flag to decide if store the csv or not
    file_path = make_path(file_path)
    os.makedirs(path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)

    return df, hm, file_path

def dfs2traversability_df(data):
    stage = th.map(df2traversability_df, data, workers=Config.WORKERS)
    return stage