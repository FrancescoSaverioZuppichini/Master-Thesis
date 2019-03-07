import glob

from postprocessing import BagsHandler, InMemoryHandler, DataFrameHandler, PatchesHandler, PostProcessingConfig

config = PostProcessingConfig(base_dir='./test',
                              maps_folder='/home/francesco/Documents/Master-Thesis/core/maps/test/',
                              # csv_dir='/home/francesco/Desktop/carino/vaevictis/data/train_no_tail#2/csv/',
                              # out_dir='/home/francesco/Desktop/data/',
                              patch_size=92,
                              advancement_th=0.12,
                              skip_every=25,
                              translation=[5, 5],
                              time_window=125,
                              name='test')

patches_h = PatchesHandler(config=config)
df_h = DataFrameHandler(successor=patches_h, config=config)
b_h = BagsHandler(config=config, successor=df_h)

bags = glob.glob('{}/**/*.bag'.format(config.bags_dir))

list(b_h(bags))