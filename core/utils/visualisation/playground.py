import pandas as pd

from utils.visualisation import show_advancement

from utils.postprocessing.utils import read_image
from utils.postprocessing.config import Config

df = pd.read_csv('/home/francesco/Desktop/carino/vaevictis/data/dataset/bars1/1548510453.5639887.csv')
hm = read_image('/home/francesco/Desktop/carino/vaevictis/data/maps/bars1.png')

show_advancement(df, hm, Config)