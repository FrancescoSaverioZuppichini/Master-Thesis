import glob
from tqdm import tqdm

from bags2csvs import *
from dfs2dataset import *
from config import Config
import time

class TraversabilityPipeplime():
    def __call__(self, bags_dir, *args, **kwargs):
        bags = glob.glob(bags_dir + '/**/*.bag')

        start = time.time()

        print('starting with {} bags'.format(len(bags)))

        stage = bags2dfs(bags)
        stage = dfs2traversability_df(stage)
        stage = traversability_dfs2patches(stage)

        result = list(x for x in tqdm(stage, total=len(bags)))

        print('processed {} bags file in {:.2f}s'.format(len(bags),
                                                        time.time() - start))

        return result

if __name__ == '__main__':
    tr_pip = TraversabilityPipeplime(Config.BAG_FOLDER)
    tr_pip(bags)

