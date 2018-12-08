import os

from utils import *
from config import Config
from pypeln import thread as th

def csv2dataset(file_path):
    map_name = filename2map(file_path)
    map_path = '{}/{}.png'.format(Config.MAPS_FOLDER, map_name)
    df = pd.read_csv(file_path)
    df = df_convert_date2timestamp(df)
    df = df_convert_quaterion2euler(df)
    df, hm = generate_single_dataset_cnn(df, map_path)

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

