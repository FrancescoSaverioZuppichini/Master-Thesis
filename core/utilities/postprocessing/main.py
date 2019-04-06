import pprint

from simulation.parser import args
from postprocessing import make_and_run_chain, PostProcessingConfig

config = PostProcessingConfig.from_args(args)

pprint.pprint(config.__dict__)

make_and_run_chain(config)
