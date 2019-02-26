import glob
import pprint
from tqdm import tqdm
from parser import args
from postprocessing import *

post_config = PostProcessingConfig.from_args(args)

pprint.pprint(post_config.__dict__)

patches_h = PatchesHandler(config=config)
df_h = DataFrameHandler(successor=patches_h, config=config)
b_h = BagsHandler(config=config, successor=df_h)

bags = glob.glob('{}/**/*.bag'.format(config.bags_dir))

list(b_h(bags))