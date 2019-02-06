from config import Config

import glob
import time

from tqdm import tqdm

from bags2csvs import *
from dfs2dataset import *

class PostProcessingPipeline():
    def __call__(self, bags_dir, *args, **kwargs):
        bags = glob.glob(bags_dir + '/**/*.bag')

        start = time.time()

        print('starting with {} bags'.format(len(bags)))

        stage = bags2dfs(bags)
        stage = dfs2traversability_df(stage)
        stage = traversability_dfs2patches(stage)

        result = list(tqdm(stage, total=len(bags)))

        print('processed {} bags file in {:.2f}s'.format(len(bags),
                                                        time.time() - start))

        return result



p_p_pip = PostProcessingPipeline()
p_p_pip(Config.BAG_FOLDER)

