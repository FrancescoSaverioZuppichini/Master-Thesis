import os
import pandas as pd
import numpy as np

from os import path

from utils import *
from config import Config
from pypeln import thread as th

P_X_KEY = 'pose__pose_position_x'
P_Y_KEY = 'pose__pose_position_y'
P_Z_KEY = 'pose__pose_position_z'

O_W_E_KEY = 'pose__pose_e_orientation_z'

def df_add_hm_coords(df, hm, sim_hm_mx_x, sim_hm_mx_y):
    df["hm_x"] = df.apply(
        lambda r: toScreenFrame(r[P_X_KEY], r[P_Y_KEY], sim_hm_mx_x, -sim_hm_mx_x, sim_hm_mx_y, -sim_hm_mx_y)[0] *
                  hm.shape[1], axis=1)
    df["hm_y"] = df.apply(
        lambda r: toScreenFrame(r[P_X_KEY], r[P_Y_KEY], sim_hm_mx_x, -sim_hm_mx_x, sim_hm_mx_y, -sim_hm_mx_y)[1] *
                  hm.shape[0], axis=1)

    return df

def df_add_advancement(df, dt):
    df = df.set_index(df.columns[0])

    df["S_oX"] = np.cos(df[O_W_E_KEY].values)
    df["S_oY"] = np.sin(df[O_W_E_KEY].values)
    assert (np.allclose(1, np.linalg.norm(df[["S_oX", "S_oY"]], axis=1)))

    df["S_dX"] = df.rolling(window=(dt + 1))[P_X_KEY].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
    df["S_dY"] = df.rolling(window=(dt + 1))[P_Y_KEY].apply(lambda x: x[-1] - x[0], raw=True).shift(-dt)
    df["S_d"] = np.linalg.norm(df[["S_dX", "S_dY"]], axis=1)

    df["advancement"] =  np.einsum('ij,ij->i', df[["S_dX", "S_dY"]], df[["S_oX", "S_oY"]])  # row-wise dot product

    return df

def df_clean_by_dropping(df):
    # skip the first two seconds and any row with nans (i.e. end of the dataset)
    df = df.loc[df.index >= 2].dropna()
    df = df.loc[df.index <= 18].dropna()  # drop also the last two seconds (if run is 20s, < 18)

    # drop the frames where the robot is upside down (orientation alpha angle [euler's angles]) to avoid false positives
    df = df.loc[df['pose__pose_e_orientation_x'] >= -2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_x'] <= 2.0].dropna()

    df = df.loc[df['pose__pose_e_orientation_y'] >= -2.0].dropna()
    df = df.loc[df['pose__pose_e_orientation_y'] <= 2.0].dropna()

    return df

def df_add_label(df, advancement_th):
    df["label"] = df["advancement"] > advancement_th
    return df

def csv2dataset(file_path):
    map_name = filename2map(file_path)
    map_path = '{}/{}.png'.format(Config.MAPS_FOLDER, map_name)
    hm  = read_image(map_path)

    df = pd.read_csv(file_path)
    df = df_convert_date2timestamp(df)
    df = df_add_dist_velocity(df)
    df = df_convert_quaterion2euler(df)
    df = df_add_hm_coords(df, hm, 5.13, 5.13)
    df = df_add_advancement(df, Config.TIME_WINDOW)
    df = df_add_label(df, Config.ADVANCEMENT_TH)
    df = df_clean_by_dropping(df)

    def make_path(file_path):
        splitted = file_path.split('/')
        map_name, file_name = splitted[-2], splitted[-1]

        return path.normpath('{}/{}/{}'.format(Config.DATASET_FOLDER, map_name, path.splitext(file_name)[0] + '.csv'))

    file_path = make_path(file_path)
    os.makedirs(path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)

def csvs2dataset(files):
    stage = th.map(csv2dataset, files, workers=Config.WORKERS)
    data = list(stage)
    return data

