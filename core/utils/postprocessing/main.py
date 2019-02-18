import glob

from config import Config
from PostProcessingPipeline import PostProcessingPipeline

post_p_pip = PostProcessingPipeline()
print(Config.BAG_FOLDER + '/**/*.bag')
post_p_pip( bags = glob.glob('/home/francesco/Desktop/carino/vaevictis/data/train_no_tail/val/bags/flat' + '/*.bag'), patches=False)
