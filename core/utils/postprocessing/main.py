import glob
from tqdm import tqdm

from bags2csvs import *
from dfs2dataset import *
from config import Config
import time

start = time.time()

bags = glob.glob(Config.BAG_FOLDER + '/**/*.bag')

print('starting with {} bags'.format(len(bags)))

stage = bags2dfs(bags)
stage = dfs2traversability_df(stage)
# TODO typo! path -> patch!!!!
stage = traversability_dfs2paths(stage)

result = list(tqdm(stage))

print('processed {} bags file in {:.2f}s'.format(len(bags),
                                                time.time() - start))