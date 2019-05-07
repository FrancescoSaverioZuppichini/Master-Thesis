import pprint
import pandas as pd
from utilities.postprocessing.postprocessing import *

import os

MODE = 'val'

PARSE_DATAFRAMES = False
ADVANCEMENT = 0.66
window = 50 * 2

DATASET_DIR = '/media/francesco/saetta/krock-dataset/{}/'.format(MODE)
N_WORKERS = 16

base_bags_dir = DATASET_DIR + '/bags/'
# DATASET_DIR = '/media/francesco/saetta/krock-dataset/test_with_obstacles/'

out_csvs_dir = DATASET_DIR + '/csvs/'
out_parsed_csvs_dir = DATASET_DIR + '/csvs_parsed/'

meta = pd.read_csv(base_bags_dir + 'meta.csv')

base_bags_dir = DATASET_DIR + '/bags/'
# meta = pd.DataFrame(data={'filename' : ['1'], 'map' : ['wall'], 'height': [1]})
# filename = meta['filename']
#
print(meta)

os.makedirs(out_csvs_dir, exist_ok=True)
os.makedirs(out_parsed_csvs_dir, exist_ok=True)

convert_bags2dfs_and_store = MultiThreadWrapper(N_WORKERS, Compose([
    # lambda x: rosbag_pandas.bag_to_dataframe(x),
    Bags2Dataframe(base_bags_dir),
    StoreDataframeKeepingSameName(out_csvs_dir)]))

read_and_parse_dfs = MultiThreadWrapper(N_WORKERS, Compose([
    ReadDataframeFilenameAndHm(out_csvs_dir, '/home/francesco/Documents/Master-Thesis/core/maps/{}/'.format(MODE)),
    ParseDataframe(),
    AddHMcoordinates(),
    CleanDataframe(lower_bound=1, offset=22),
    drop_uselesss_columns,
    StoreDataframeKeepingSameName(out_parsed_csvs_dir)
]))

if PARSE_DATAFRAMES:
    dfs_from_bags = convert_bags2dfs_and_store( meta['filename'])
    parsed_dfs = read_and_parse_dfs(meta.iterrows())
#

# #
ADVANCEMENT = 0.66
# PATCH_SIZE = 50 * 2 + 2 * 7
window = 50 * 2

print('[INFO] window={}'.format(window))
patches_out_dir = DATASET_DIR + '/patches/{}/'.format(ADVANCEMENT)
meta_df_out_dir = DATASET_DIR + '/csvs_patches/'

os.makedirs(patches_out_dir, exist_ok=True)
os.makedirs(meta_df_out_dir, exist_ok=True)

extract_patches = MultiThreadWrapper(N_WORKERS, Compose([
    ReadDataframeFilenameAndHm(out_parsed_csvs_dir,
                               '/home/francesco/Documents/Master-Thesis/core/maps/{}/'.format(MODE)),
    AddAdvancement(window),
    ExtractPatches(patch_extract_stategy=KrockPatchExtractStrategy(max_advancement=ADVANCEMENT)),
    StorePatches(patches_out_dir, meta_df_out_dir)

]))

extract_patches(meta.iterrows())
