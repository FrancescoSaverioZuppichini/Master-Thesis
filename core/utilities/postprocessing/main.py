from utilities.visualisation import DataFrameVisualization
from utilities.postprocessing.PostProcessing import PostProcessing

vis = DataFrameVisualization.from_root('/media/francesco/saetta/krock-dataset/val/csvs/')
vis.df = vis.df[vis.df['advancement'] >= -0.2]
vis(tr=0.2)