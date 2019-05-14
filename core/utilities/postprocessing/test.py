import glob
from tqdm import tqdm
import time

from pypeln import thread as th
from utilities.postprocessing.handlers import BagsHandler, InMemoryHandler, DataFrameHandler, PatchesHandler, PostProcessingConfig, MultiThreadWrapper

config = PostProcessingConfig(base_dir='./test/',
                              maps_folder='/home/francesco/Documents/Master-Thesis/core/maps/test/',
                              # csv_dir='/home/francesco/Desktop/carino/vaevictis/data/train_no_tail#2/csv/',
                              out_dir='./test/',
                              patch_size=92,
                              advancement_th=0.12,
                              skip_every=12,
                              translation=[5, 5],
                              time_window=125,
                              name='test')

patches_h = PatchesHandler(config=config, debug=False)
df_h = DataFrameHandler(successor=patches_h, config=config)
b_h = BagsHandler(config=config, successor=df_h)

files = glob.glob('{}/csvs/**/*.csv'.format(config.base_dir))

if len(files) > 0:
    pip = InMemoryHandler(config=config, successor=patches_h)
else:
    files = glob.glob('{}/**/*.bag'.format(config.bags_dir))
    pip = b_h


th_wrap = MultiThreadWrapper(2)
data = th_wrap(pip, files)
