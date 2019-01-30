import glob

from bags2csvs import *
from dfs2dataset import *
from config import Config


bags = glob.glob(Config.BAG_FOLDER + '/flat/*.bag')
# bags = glob.glob(Config.BAG_FOLDER + '/querry-big/*.bag')
# bags = glob.glob('/home/francesco/Desktop/carino/vaevictis/data/bags/querry-big/*.bag')

stage = bags2dfs(bags)
stage = dfs2traversability_df(stage)
# stage = traversability_dfs2paths(stage)

list(stage)
