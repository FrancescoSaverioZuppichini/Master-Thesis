import glob

from bags2csvs import *
from dfs2dataset import *
from config import Config

bags = glob.glob(Config.BAG_FOLDER + '/flat/*.bag')
stage = bags2dfs(bags)
stage = dfs2traversability_df(stage)
stage = traversability_dfs2paths(stage)

list(stage)