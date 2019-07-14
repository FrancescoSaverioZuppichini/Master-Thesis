from utilities.visualisation import DataFrameVisualization
from utilities.postprocessing.PostProcessing import PostProcessing
# post = PostProcessing(root='/media/francesco/saetta/krock-dataset/new-test-random',
#                       maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/test/',
#                       advancement=0.7,
#                       time_window=100)
#
# post()
# #
# post = PostProcessing(root='/media/francesco/saetta/krock-dataset/train',
#                       maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/new-train/',
#                       advancement=1,
#                       time_window=150)
#
# post()
#

post = PostProcessing(root='/media/francesco/saetta/krock-dataset/new-new-val',
                      maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/val/',
                      advancement=0.7,
                      time_window=100)

post()



# post = PostProcessing(root='/media/francesco/saetta/krock-dataset/crop/',
#                       maps_dir='/home/francesco/Documents/Master-Thesis/core/maps/new-train/',
#                       advancement=0.7,
#                       time_window=100)
#
# post()
vis = DataFrameVisualization.from_root('/media/francesco/saetta/krock-dataset/tr/slope_rocks1/150/csvs')
# vis.df = vis.df[vis.df['advancement'] >= -0.2]
vis(0.2)