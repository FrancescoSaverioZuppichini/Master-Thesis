import pprint

from utilities.postprocessing.parser import args
from utilities.postprocessing.postprocessing import make_and_run_chain, PostProcessingConfig

config = PostProcessingConfig.from_args(args)

pprint.pprint(config.__dict__)

make_and_run_chain(config)
