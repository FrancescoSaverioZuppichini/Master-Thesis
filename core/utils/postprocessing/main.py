import glob

from bags2csvs import *
from csvs2dataset import *
from config import Config

bags = glob.glob(Config.BAG_FOLDER + '/flat/*.bag')
stage = bags2dfs(bags)
stage = csvs2dataset(stage)
stage = csvs2paths(stage)

list(stage)