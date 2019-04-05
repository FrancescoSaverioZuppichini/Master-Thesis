import pandas as pd

from utils.visualisation import *

from utils.postprocessing.utils import read_image

df = pd.read_csv('/media/francesco/saetta/no-shift-88-750/train/df/bars1/1550614988.2771952-patch.csv')
hm = read_image('/home/francesco/Desktop/carino/vaevictis/data/maps/bars1.png')

#
# show_trajectory(df.iterrows(), hm)