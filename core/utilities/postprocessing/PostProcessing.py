from utilities.postprocessing.handlers import *
from utilities.postprocessing.utils import TraversabilityDir
from utilities.postprocessing.utils import KrockPatchExtractStrategy, PatchExtractStrategy


class PostProcessing():
    def __init__(self, root, maps_dir, advancement, time_window):
        self.dir = TraversabilityDir(root, maps_dir, advancement)
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
            # AddAdvancement(time_window),
            ExtractPatches(patch_extract_stategy=KrockPatchExtractStrategy(max_advancement=self.advancement)),
            StorePatches(self.dir.patches_dir, self.dir.csvs_patches_dir)

        ]))

    def __call__(self):
        meta = self.dir.meta
        if self.dir.should_convert_bags:
            self.convert_bags2dfs_and_store(meta['filename'])
            self.read_and_parse_dfs(meta.iterrows())
        self.extract_patches(meta.iterrows())

    @classmethod
    def from_args(cls, args):
        return cls(args.root, args.map_dir, args.advancement, args.time_window)

