import glob

from utils import *

DATA_FOLDER = '/home/francesco/Desktop/carino/vaevictis/data/'
files = glob.glob(DATA_FOLDER + '/*/**.csv')



map_name = filename2map(files[0])
map_path = '/home/francesco/Documents/Master-Thesis/core/maps/{}.png'.format(map_name)

dff, hm = generate_single_dataset_cnn(files[0], map_path)


dff.to_csv('./test.csv')
print(dff)