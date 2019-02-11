import glob
from tqdm import tqdm

from bags2csvs import *
from dfs2dataset import *
from config import Config
import time


class PostProcessingPipeline():
    def __call__(self, bags_dir):
        start = time.time()

        bags = glob.glob(Config.BAG_FOLDER + '/**/*.bag')

        print('starting with {} bags in {}'.format(len(bags), Config.BAG_FOLDER))

        stage = bags2dfs(bags)
        stage = dfs2traversability_df(stage)
        stage = traversability_dfs2patches(stage)

        result = list(tqdm(stage, total=len(bags)))

        print('processed {} bags file in {:.2f}s'.format(len(bags),
                                                        time.time() - start))

        return result

if __name__ == '__main__':
    post_p_pip = PostProcessingPipeline()
    post_p_pip(Config.BAG_FOLDER)
