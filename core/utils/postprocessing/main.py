import glob
import pprint
from tqdm import tqdm
from parser import args
from postprocessing import PostProcessingConfig, BagsPostProcessing, \
    DataFramePostProcessing, DataFrame2PatchesPostProcessing

post_config = PostProcessingConfig.from_args(args)

pprint.pprint(post_config.__dict__)

bags_post = BagsPostProcessing(post_config)
df_post = DataFramePostProcessing(post_config)
df2patches_post = DataFrame2PatchesPostProcessing(post_config)

bags = glob.glob('{}/**/*.bag'.format(post_config.bags_dir))

print(bags)

stage = df2patches_post(df_post(bags_post(bags)))

list(tqdm(stage, total=len(bags)))