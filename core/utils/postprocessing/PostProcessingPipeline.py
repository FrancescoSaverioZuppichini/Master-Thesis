import glob
from tqdm import tqdm

from bags2csvs import *
from dfs2dataset import *
from config import Config
import time


class PostProcessingPipeline():
    def __call__(self, bags, verbose=True, patches=True):
        start = time.time()

        if verbose : print('starting with {} bags in {}'.format(len(bags), Config.BAG_FOLDER))

        stage = bags2dfs(bags)
        stage = dfs2traversability_df(stage)
        if patches: stage = traversability_dfs2patches(stage)

        result = list(tqdm(stage, total=len(bags)))

        if verbose: print('processed {} bags file in {:.2f}s'.format(len(bags),
                                                        time.time() - start))

        return result