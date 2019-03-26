import glob
import pprint

from art import *
from tqdm import tqdm
from parser import args, utility_args
from postprocessing import make_and_run_chain, PostProcessingConfig

config = PostProcessingConfig.from_args(args)

pprint.pprint(config.__dict__)

make_and_run_chain(config)
