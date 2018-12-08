import glob

import matplotlib.pyplot as plt

from utils import *

DATA_FOLDER = '/home/francesco/Desktop/carino/vaevictis/data/'
files = glob.glob(DATA_FOLDER + '/bumps1/**.csv')



map_name = filename2map(files[0])
map_path = '/home/francesco/Documents/Master-Thesis/core/maps/{}.png'.format(map_name)


for file in files:
    df = pd.read_csv(file)
    df = df_convert2timestamp(df)
    df = df_convert_quaterion2euler(df)

    df, hm = generate_single_dataset_cnn(df, map_path)
    df.to_csv('./test.csv')
