import rosbag_pandas

from utilities.pipeline import Handler
from .functional import *


class StoreDataframeKeepingSameName():
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def __call__(self, data):
        df, hm, filename = data
        store_data_keeping_same_name(df, filename, self.out_dir)
        return df, hm, filename


class Bags2Dataframe(Handler):
    def __init__(self, base_dir, out_dir=None):
        self.base_dir = base_dir

    def __call__(self, filename):
        df = rosbag_pandas.bag_to_dataframe(self.base_dir + '/' + filename + '.bag')
        return df, None, filename


class ReadDataframeFilenameAndHm():
    def __init__(self, base_dir, hm_dir):
        self.base_dir = base_dir
        self.hm_dir = hm_dir

    def __call__(self, data):
        idx, row = data
        df, hm = open_df_and_hm_from_meta_row(row, self.base_dir, self.hm_dir)
        df['height'] = row['height']
        return df, hm, row['filename']


class ParseDataframe(Handler):
    def __call__(self, data):
        df, hm, filename = data
        df = parse_dataframe(df)
        return df, hm, filename


class AddAdvancement(Handler):
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, data):
        df, hm, filename = data
        if len(df) > 0: df = add_advancement(df, self.dt)
        return df, hm, filename


class CleanDataframe(Handler):
    def __init__(self, lower_bound, offset=0):
        self.lower_bound = lower_bound
        self.offset = offset

    def __call__(self, data):
        df, hm, filename = data
        df = clean_dataframe(df, hm, self.lower_bound, self.offset)
        return df, hm, filename


class AddHMcoordinates(Handler):
    def __init__(self, resolution=0.02, translation=[5, 5]):
        self.resolution = resolution
        self.translation = translation

    def __call__(self, data):
        df, hm, filename = data
        df = add_hm_coordinates2df(df, hm, self.resolution, self.translation)
        return df, hm, filename


class ReadHm(Handler):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def __call__(self, name):
        return read_image(self.base_dir + name + '.png')


class ReadDataframeAndStoreName():
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def __call__(self, path):
        df = pd.read_csv(self.base_dir + '/' + path + '.csv')
        return df


class ExtractPatches():
    def __init__(self, patch_extract_stategy):
        self.patch_extract_stategy = patch_extract_stategy

    def __call__(self, data):
        df, hm, filename = data
        patches = extract_patches(df, hm, self.patch_extract_stategy)

        return df, patches, filename


class StorePatches():
    def __init__(self, out_dir, meta_df_out_dir):
        self.out_dir = out_dir
        self.meta_df_out_dir = meta_df_out_dir

    def __call__(self, data):
        df, patches, filename = data
        df, filename = store_patches(df, filename, patches, self.meta_df_out_dir, self.out_dir)

        return df, filename
