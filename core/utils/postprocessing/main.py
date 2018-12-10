import glob

from config import Config
from bags2csvs import bags2csvs
from csvs2dataset import csvs2dataset

bags = glob.glob(Config.BAG_FOLDER + '/flat/**.bag')
bags2csvs(bags)


csvs = glob.glob(Config.CSV_FOLDER + '/flat/**.csv')
csvs2dataset(csvs)
