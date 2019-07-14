from utilities.postprocessing.handlers import *
from utilities.postprocessing.utils import TraversabilityDir
from utilities.postprocessing.extractions import KrockPatchExtractStrategy
from utilities.pipeline import Compose, MultiThreadWrapper

class PostProcessing():
    """
    This class parse the stored bags and create the final dataset. It uses TraversabilityDir to understand in which folder
    each stage must be stored.
    """
    def __init__(self, root, maps_dir, advancement, time_window):

        self.dir = TraversabilityDir(root, maps_dir, time_window)
        self.advancement, self.time_window = advancement, time_window

        self.convert_bags2dfs_and_store = MultiThreadWrapper(16, Compose([
            Bags2Dataframe(self.dir.bags_dir),
            StoreDataframeKeepingSameName(self.dir.csvs_dir)]))

        self.read_and_parse_dfs = MultiThreadWrapper(16, Compose([
            ReadDataframeFilenameAndHm(self.dir.csvs_dir,
                                       self.dir.maps_dir),
            ParseDataframe(),
            AddHMcoordinates(),
            CleanDataframe(lower_bound=1, offset=22),
            StoreDataframeKeepingSameName(self.dir.csvs_parsed_dir)
        ]))

        self.extract_patches = MultiThreadWrapper(16, Compose([
            ReadDataframeFilenameAndHm(self.dir.csvs_parsed_dir,
                                       self.dir.maps_dir),
            ExtractPatches(patch_extract_stategy=KrockPatchExtractStrategy(max_advancement=self.advancement)),
            StorePatches(self.dir.patches_dir),
            AddAdvancement(time_window),
            StoreDataframePatches(self.dir.csvs_patches_dir)
        ]))

    def __call__(self):
        meta = self.dir.meta
        # convert the bags file is expensive, we only have to do it once
        if self.dir.should_convert_bags:
            self.convert_bags2dfs_and_store(meta['filename'])
            self.read_and_parse_dfs(meta.iterrows())
        self.extract_patches(meta.iterrows())

    @classmethod
    def from_args(cls, args):
        return cls(args.root, args.map_dir, args.advancement, args.time_window)

