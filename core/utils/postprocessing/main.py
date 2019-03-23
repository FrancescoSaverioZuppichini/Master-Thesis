import glob
import pprint

from art import *
from tqdm import tqdm
from parser import args, utility_args
from postprocessing import BagsHandler, InMemoryHandler, DataFrameHandler, PatchesHandler, PostProcessingConfig

config = PostProcessingConfig.from_args(args)

pprint.pprint(config.__dict__)

patches_h = PatchesHandler(config=config)
df_h = DataFrameHandler(successor=patches_h, config=config)
b_h = BagsHandler(config=config, successor=df_h)

bags = glob.glob('{}/**/*.bag'.format(config.bags_dir))

print(len(bags))
list(b_h(bags))