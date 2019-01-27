import glob

from bags2csvs import *
from csvs2dataset import *
from config import Config

# bags = glob.glob(Config.BAG_FOLDER + '/*/**.bag')
# bags2csvs(bags)
csvs = glob.glob(Config.CSV_FOLDER + '/bars1/1548510453.5639887.csv')
# csvs = glob.glob(Config.CSV_FOLDER + '/flat/**.csv')
# csvs = glob.glob(Config.CSV_FOLDER + '/*/**.csv')
data = csvs2dataset(csvs)

df2paths(data[0])