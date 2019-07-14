import os
import pandas as pd
import glob
from os import path

class TraversabilityDir():
    def __init__(self, root, maps_dir, time_window):
        self.root, self.maps_dir, self.time_window = root, maps_dir, time_window

    @property
    def meta(self):
        return pd.read_csv(self.root + '/meta.csv')

    @property
    def bags_dir(self):
        return path.normpath(self.root + '/bags')

    @property
    def csvs_dir(self):
        dir = path.normpath(self.root + '/csvs')
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def csvs_parsed_dir(self):
        dir = path.normpath(self.root + '/csvs')
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def csvs_patches_dir(self):
        dir = path.normpath(self.root + '/{}/csvs'.format(self.time_window))
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def patches_dir(self):
        dir = path.normpath(self.root + '/{}/patches/'.format(self.time_window))
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def should_convert_bags(self):
        return len(glob.glob(self.csvs_parsed_dir + '/*.csv')) <= 0

    def should_convert2bag(self, filename):
        return path.exists(self.csvs_parsed_dir + '/' + filename + '.csv')

    def exist(self, filename):
        return path.exists(self.csvs_patches_dir + '/' + filename + '.csv')
