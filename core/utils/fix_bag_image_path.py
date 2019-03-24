import pandas as pd
from pypeln import thread as th
from os import path
import tqdm
import glob

def fix_path(x):
    name = path.basename(x)
    return '/patches/{}'.format(name)

csv_paths = glob.glob('/media/francesco/saetta/85-750/test/df/**/*.csv')

def fix_df(csv_path):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image_path'].apply(fix_path)
    df.to_csv(csv_path)

stage = th.map(fix_df, csv_paths, workers=16)
list(tqdm.tqdm(stage, total=len(csv_paths)))