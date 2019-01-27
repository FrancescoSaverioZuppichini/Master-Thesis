import glob

from bags2csvs import *
from postprocessing import csvs2dataset
from config import Config

bags = glob.glob(Config.BAG_FOLDER + '/*/**.bag')
bags2csvs(bags)

# csvs = glob.glob(Config.CSV_FOLDER + '/flat/1542573200.115245.csv')
# csvs = glob.glob(Config.CSV_FOLDER + '/flat/**.csv')
csvs = glob.glob(Config.CSV_FOLDER + '/*/**.csv')
csvs2dataset(csvs)
