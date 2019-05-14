from utilities.visualisation import DataFrameVisualization
from utilities.postprocessing.PostProcessing import PostProcessing

post = PostProcessing(root='/media/francesco/saetta/krock-dataset/val/',
                      maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/val/',
                      advancement=0.66,
                      time_window=50 * 3)

post()

vis = DataFrameVisualization.from_root(post.dir.csvs_patches_dir)
vis(tr=0.2)
