import glob

from config import Config
from PostProcessingPipeline import PostProcessingPipeline

post_p_pip = PostProcessingPipeline()
post_p_pip(bags = glob.glob(Config.BAG_FOLDER + '/**/*.bag'))
