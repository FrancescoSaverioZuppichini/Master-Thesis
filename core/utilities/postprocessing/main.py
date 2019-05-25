from utilities.visualisation import DataFrameVisualization
from utilities.postprocessing.PostProcessing import PostProcessing
post = PostProcessing(root='/media/francesco/saetta/krock-dataset/new-test-random',
                      maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/test/',
                      advancement=0.71,
                      time_window=100)

post()

post = PostProcessing(root='/media/francesco/saetta/krock-dataset/train',
                      maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/new-train/',
                      advancement=0.71,
                      time_window=100)

post()


post = PostProcessing(root='/media/francesco/saetta/krock-dataset/new-new-val',
                      maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/val/',
                      advancement=0.71,
                      time_window=100)

post()

# vis = DataFrameVisualization.from_root('/media/francesco/saetta/krock-dataset/new-train/csvs/')
# vis.df = vis.df[vis.df['advancement'] >= -0.2]
# vis(tr=0.2)
